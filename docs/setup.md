# 환경 세팅 기록

## 스택

| 영역 | 선택 | 버전 |
|---|---|---|
| 패키지 관리 | uv | 0.11.32 |
| Python | CPython | **3.12.13** (고정) |
| 백엔드 | FastAPI | 0.141.1 |
| 설정 | pydantic-settings | 2.15.0 |
| LLM | google-genai | 2.17.0 |
| 프론트 | Next.js / React | 16.3.0 / 19.2.8 |
| DB | pgvector/pgvector | pg18 |

---

## Python 3.12 전환

`uv init`이 시스템 기본값(3.14.6)으로 프로젝트를 만들어서 3.12로 되돌린 기록.

### 겪은 에러

```
> uv python pin 3.12
error: The requested Python version `3.12` is incompatible with
       the project `requires-python` value of `>=3.14`.
```

`uv python pin`은 핀을 쓰기 전에 `pyproject.toml`의 `requires-python`과 충돌하는지 먼저 검사한다.
**순서가 중요하다 — pyproject를 먼저 고쳐야 pin이 통과한다.**

### 올바른 순서

```powershell
# 1. pyproject.toml 수정
#    requires-python = ">=3.14"  →  ">=3.12,<3.13"
# 2. 핀 변경
uv python pin 3.12        # .python-version 갱신
# 3. 가상환경 생성
uv sync
uv run python -V          # Python 3.12.13
```

### `<3.13` 상한을 둔 이유

라이브러리가 아니라 애플리케이션이므로 3.12에서만 돌면 된다.
`>=3.12`만 쓰면 uv가 3.12~3.15 전부에서 동작하는 조합을 찾으려 해서,
torch 같은 패키지가 불필요하게 낮은 버전으로 잡힐 수 있다.

### 고정 이유 (README에도 기록)

3.14 전용 기능이 필요 없고, Windows/CUDA 환경에서 torch·numpy 등 ML 패키지의
wheel 지원이 안정적인 버전이 3.12.

---

## 백엔드 구조 (MVC2)

```
backend/
├── main.py                 # FastAPI 인스턴스 + CORS + 라우터 자동 등록
├── .env.example            # GEMINI_API_KEY, CORS_ORIGINS
├── .env                    # 직접 생성 필요 (gitignore됨)
└── src/
    ├── core/settings.py            # pydantic-settings
    ├── controllers/chat_controller.py
    ├── services/chat_service.py
    ├── repositories/document_repository.py
    ├── models/document.py          # DB 라이브러리 미설치, 주석 예시만
    └── schemas/chat.py
```

`__init__.py`는 두지 않는다. Python 3.3+ namespace package로 import가 정상 동작한다.

### 라우터 자동 등록

`main.py`의 `register_routers()`가 `src/controllers` 아래 모듈을 훑어 `router`를 찾아
`app.include_router()`로 등록한다. 컨트롤러 파일을 추가하는 것만으로 엔드포인트가 붙고
`main.py`는 고칠 일이 없다.

**자동화하는 것은 '등록'뿐이다.** 경로와 태그는 각 컨트롤러가 소유한다.

```python
router = APIRouter(prefix="/chat", tags=["chat"])
```

파일명에서 prefix를 유도하지 않는 이유는, 그렇게 하면 파일 rename이 곧 API 주소 변경이
되고 `/v1/...` 같은 예외 경로를 줄 수 없기 때문이다. 전체 URL 목록은
`grep -rn "APIRouter(" src/controllers`로 한 번에 확인할 수 있다.

| 규칙 | 이유 |
|---|---|
| prefix/tags는 `APIRouter()`에 쓴다 | 주소의 출처를 컨트롤러 한 곳으로 모은다 |
| `router`가 없는 모듈은 기동 실패 | 오타로 엔드포인트가 사라져도 서버는 뜨는 사고를 막는다 |
| 컨트롤러가 아닌 파일은 `_` 로 시작 | 자동 등록 대상에서 제외된다 |
| 경로가 겹치는 라우트는 같은 파일에 둔다 | 파일 간 등록 순서는 보장되지 않는다 |

### 계층 구분 기준 — "무엇을 모르는가"

| 계층 | 아는 것 | **모르는 것** |
|---|---|---|
| controller | URL, HTTP 상태코드, 요청/응답 형식 | 비즈니스 로직, SQL |
| service | 처리 순서, 도메인 규칙 | **HTTP** (`Request`, `HTTPException` 안 씀) |
| repository | 쿼리, 테이블 | 비즈니스 규칙, HTTP |

service가 HTTP를 모르게 유지하면, 나중에 "AI Hub 데이터 일괄 임베딩" 같은
배치 스크립트에서 같은 서비스를 그대로 재사용할 수 있다.

판단이 애매할 때:
- "이 코드를 CLI에서도 부를 수 있나?" → 없으면 controller로 갈 코드
- "DB를 다른 걸로 바꿔도 그대로인가?" → 그러면 service, 아니면 repository
- controller 함수가 5줄을 넘으면 service로 갈 로직이 섞인 것

### models vs schemas

| | `models/` | `schemas/` |
|---|---|---|
| 대상 | **DB** | **프론트(외부)** |
| 라이브러리 | SQLAlchemy | Pydantic |
| 바뀌는 이유 | 저장 구조 변경 | API 스펙 변경 |

이 프로젝트 예시:
- `models.Document`에는 `embedding` 벡터 768개가 있다 (검색에 필요)
- `schemas.ChatResponse`에는 없다 (프론트에 내려줄 이유가 없음)
- `ChatRequest.top_k`는 요청 옵션일 뿐 DB에 저장할 대상이 아니다

합치면 DB 컬럼 추가가 곧바로 API 응답 노출로 이어진다. 분리해두면
"내보낼 것만 schema에 적는다"가 강제된다.

### core vs utils

- `core/` — 앱이 돌아가기 위한 기반 (설정, DB 연결, 보안)
- `utils/` — 어디서든 쓰는 순수 함수 (텍스트 정제, 청킹 등). 필요해지면 추가

### CORS

프론트(`localhost:3000`)와 백엔드(`localhost:8000`)는 포트가 달라 브라우저 기준
**서로 다른 오리진**이다. 오리진은 스킴+호스트+포트가 전부 일치해야 같은 것으로 친다.
`main.py`에서 `CORSMiddleware`로 허용 오리진을 지정한다.

별도 라이브러리 설치는 필요 없다. Starlette 내장이라 `fastapi.middleware.cors`에서
바로 import한다 (Flask의 `flask-cors` 같은 패키지가 FastAPI에는 불필요).

허용 오리진은 `Settings.CORS_ORIGINS`로 분리해 `.env`에서 주입한다.

```
CORS_ORIGINS='["http://localhost:3000"]'
```

**JSON 배열로 써야 한다.** pydantic-settings가 `list[str]` 필드를 JSON으로 파싱하므로
`CORS_ORIGINS=http://localhost:3000`처럼 쓰면 기동 시점에 파싱 에러로 죽는다.
겉따옴표는 dotenv가 벗겨준다.

#### 걸리기 쉬운 지점

- **`allow_methods`를 빠뜨리면 안 된다.** Starlette 기본값이 `("GET",)`이라
  `POST /chat`의 preflight가 `400 Disallowed CORS method`로 막힌다.
  `Content-Type: application/json`은 simple request가 아니라 preflight가 항상 뜬다.
- `127.0.0.1:3000`은 `localhost:3000`과 **다른 오리진**이다. 끝에 `/`를 붙여도 매칭 실패.
- `allow_credentials=True`와 `allow_origins=["*"]`는 함께 쓰지 않는다.
  기본값을 `["*"]`로 두면 .env를 빠뜨렸을 때 조용히 전체 개방되므로,
  기본값은 로컬 오리진으로 좁혀 두고 배포 도메인은 `.env`로 주입한다.
- 브라우저 콘솔의 "CORS 에러"가 항상 CORS 문제인 것은 아니다. 미들웨어를 타기 전에
  죽는 예외에는 CORS 헤더가 안 붙어서 실제 원인(500 등)이 가려진다. Network 탭에서
  상태코드부터 볼 것.

#### 배포 시 (메인 프로젝트)

프론트 `daengs.weareithero.cloud` / 백엔드 `daengback.weareithero.cloud`처럼
도메인을 나눠도 서브도메인이 다르면 cross-origin이므로 미들웨어는 그대로 필요하다.

다만 등록 도메인(`weareithero.cloud`)이 같아 **쿠키 기준으로는 same-site**다.
인증 쿠키를 `SameSite=Lax` + `Domain=.weareithero.cloud`로 두 서브도메인이 공유할 수
있다 (완전히 다른 도메인 두 개를 쓸 때보다 유리한 점).

CORS는 보안 장치가 아니다. 브라우저에만 적용되므로 curl이나 서버 간 요청에는
아무 제약이 걸리지 않는다. 실제 방어는 인증·인가에서 한다.

### 동작 확인

```
GET  /      → 200
POST /chat  → 200 (더미 응답 + sources)
POST /chat  → 422 (question='' 자동 검증 실패)
```

`.env`가 없으면 기동 즉시 `GEMINI_API_KEY Field required`로 죽는다. 의도한 동작 —
API 호출 시점에 `None`으로 터지는 것보다 낫다.

---

## Docker / pgvector

### ⚠️ PG18 볼륨 경로 함정

```
Config.Volumes: {"/var/lib/postgresql": {}}
PGDATA=/var/lib/postgresql/18/docker
```

PostgreSQL 18 이미지부터 데이터 디렉터리가 바뀌었다.
예전 `/var/lib/postgresql/data` → 현재 `/var/lib/postgresql/18/docker`.

인터넷 예제에 흔한 `pgdata:/var/lib/postgresql/data`를 그대로 쓰면
**엉뚱한 빈 디렉터리를 마운트**하는 셈이라, 컨테이너는 정상 기동하지만
`docker compose down -v`나 컨테이너 재생성 시 데이터가 전부 날아간다.

→ `compose.yaml`에서 상위 경로 `pgdata:/var/lib/postgresql`로 마운트했다.

### 그 외 결정

- **볼륨 이름 고정** (`name: rag-study-pgdata`) — 안 하면 폴더명 기반 접두사가 붙어서
  폴더 이름을 바꾸면 볼륨을 잃는다
- **`db/init/`** — `docker-entrypoint-initdb.d`는 볼륨이 비어 있을 때만 1회 실행된다.
  이미 만든 DB에 SQL을 추가해도 실행되지 않으므로, 스키마 변경은 나중에 alembic으로
- **`.env` 위치 주의** — compose의 `${POSTGRES_USER}`는 **루트 `.env`**,
  pydantic-settings는 **`backend/.env`**를 읽는다. 서로 다른 파일이다

```powershell
docker compose up -d
docker compose exec db psql -U postgres -d ragdb -c "\dx"   # vector 확장 확인
```

---

## git 설정

### `.gitignore`

- `data/` 제외 — AI Hub 원본 27MB, **재배포 제약 라이선스**라 커밋 금지
- `.env`, `.venv/`, `__pycache__/` 제외
- **겪은 사고**: `models/` 규칙이 `backend/src/models/`까지 잡아먹었다.
  gitignore는 슬래시 없는 패턴을 모든 하위 디렉터리에 적용한다.
  → `/models/`로 루트 고정해서 해결

### `.gitattributes`

```gitattributes
* text=auto eol=lf
uv.lock            linguist-generated=true
package-lock.json  linguist-generated=true
```

Windows/Mac 혼용 시 줄바꿈 때문에 diff가 통째로 바뀌는 것을 방지한다.
개인 git 설정과 무관하게 저장소에 커밋되어 모두에게 적용되는 것이 핵심.

`core.autocrlf=true` 덕에 이미 LF로 저장돼 있어서 `git add --renormalize .` 결과
변경 파일이 0건이었다 — 정규화 커밋 없이 끝났다.

**팀원 전달사항**: `.gitattributes`는 이미 클론한 사람에게 자동 적용되지 않는다.
`git pull` 후 각자 `git add --renormalize .` 한 번 실행.

### 커밋 규칙

- `uv.lock` **반드시 커밋** (팀원이 동일 버전 조합을 받게 함)
- `.venv`는 커밋 금지

---

## 참고: websockets 버전이 내려간 이유

`uv add google-genai` 시 `websockets 17.0.1 → 16.1.1` 다운그레이드가 일어났다.

```
google-genai      → websockets>=13.0.0,<17.0   ← 상한 있음
uvicorn[standard] → websockets>=13.0           ← 상한 없음
```

uv는 프로젝트 전체에서 패키지당 버전을 하나만 유지한다(단일 해결 그래프).
두 조건의 교집합 `>=13.0,<17.0`에서 최고 버전이 16.1.1.
uvicorn은 `>=13.0`만 요구하므로 문제 없다. 정상 동작이다.

조건이 아예 안 겹치면(A가 `<17`, B가 `>=17`) uv는 설치를 거부하고 충돌 에러를 낸다.
