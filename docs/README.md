# docs

`rag-study` 프로젝트 진행 기록. 세션이 끊겨도 여기만 읽으면 맥락을 복구할 수 있도록 정리한다.

| 문서 | 내용 |
|---|---|
| [setup.md](setup.md) | 환경 세팅 (Python 3.12, uv, 백엔드 구조, Docker, git 설정) |
| [dataset-analysis.md](dataset-analysis.md) | AI Hub 데이터셋 실측 분석 결과 |
| [rag-design.md](rag-design.md) | RAG 설계 결정과 그 근거 |

## 현재 상태 (2026-08-13 기준)

- 백엔드 / 프론트 / DB 기초 세팅 완료, `main`에 푸시됨 (`c8001cb`)
- 데이터 분석 완료 — [dataset-analysis.md](dataset-analysis.md)
- **RAG 재설계 완료, 구현 착수 직전** — 설계는 [rag-design.md](rag-design.md)

### 코드의 실제 상태 (중요)

문서에 "1차 시도 완료"로 적혀 있던 적재·검색 코드는 **저장소에 없다.**
다른 세션에서 애드혹으로 돌린 것이라 남지 않았다. 현재 백엔드는 골격뿐이다.

| 파일 | 상태 |
|---|---|
| `services/chat_service.py` | `_embed()`가 0벡터 768개 반환, `_generate()` 임시 문자열 |
| `repositories/document_repository.py` | 하드코딩 더미 2건 |
| `models/document.py` | 전부 주석. 내용도 설계와 불일치(`content`/`category`) |
| `pyproject.toml` | sqlalchemy · pgvector · 임베딩 런타임 없음 |
| `db/init/` | `CREATE EXTENSION vector` 뿐, 테이블 DDL 없음 |

### 데이터 재검증 (2026-08-13)

zip 18개를 다시 파싱해 [dataset-analysis.md](dataset-analysis.md) 수치와 대조 —
**QA 21,606건(Training 19,206 / Validation 2,400), 무결성, 메타데이터, 길이 전부 일치.**
Train↔Valid 질문 누수 1건도 그대로라 평가셋 전제가 유효하다.

`가격/비용 언급`(648→663)과 `input 질문형태`(17,646→18,615)만 다른데, 데이터가
아니라 정규식 기준 차이다. 비율 판정은 안 바뀌므로 문서를 고칠 필요 없다.

### 백엔드 골격 (08-13)

RAG 본작업과는 별개로, 메인 프로젝트에 그대로 옮길 골격을 정리한 기록.
판단 근거는 [setup.md](setup.md)에 있다.

- CORS 미들웨어 추가 (`461ab31`) — 프론트에서 백엔드를 직접 호출 가능해졌다.
  메인 프로젝트도 프론트/백엔드에 각각 도메인을 줄 예정이라 같은 설정이 필요하다.
- 컨트롤러 라우터 자동 등록 (`f4280c4`) — 컨트롤러 파일 추가만으로 엔드포인트가 붙는다.
  prefix/tags는 각 `APIRouter`가 계속 소유한다.

## 다음 할 일

상세 근거는 전부 [rag-design.md](rag-design.md)에 있다. 여기는 순서만.

### 진행 중인 것부터 — 내일 여기서 시작

**1. `scripts/prepare_data.py`** — zip → JSONL

- QA 21,606건: `utf-8-sig`로 읽을 것 (BOM 때문에 첫 키가 `meta`로 안 잡힌다)
- 원천 239건: **전처리(각주번호·표캡션·영어문단 제거) → 청킹** 순서
- 출력은 `content` / `payload` / `source` / 메타데이터 형태로 (스키마는 rag-design 참고)
- 파싱 로직은 **`scripts/verify_data.py`에 이미 검증된 게 있다.** zip을 풀지 않고
  그대로 읽어 21,606건을 파싱하므로 여기서 잘라 쓰면 된다
  (`uv run python ../scripts/verify_data.py`로 언제든 데이터 상태 재확인 가능)

**2. 구어체 변형셋** — Validation 2,400건 → 지표 1 질의셋.
HF `Sr523/big-red-bark-chat-evaluation` 질문 100개를 훑어 few-shot 예시로 쓸 것

**3~6.** 모델 3개 임베딩(Colab) → numpy로 지표 1 비교 → 모델·차원 확정 →
DB 적재 → 리랭커 → 원천데이터 A/B → 서비스 연결

### 막힌 것 — 시작 전에 확인 필요

- **GPU VRAM이 4GB인지 8GB인지.** 임베딩(~2GB) + 리랭커(~2GB) 동시 로드가
  가능한지가 갈린다. Colab을 쓰더라도 리랭커는 로컬 상주라 이게 결정적

### 계류 중인 수정 (하기로 정했으나 아직 손 안 댐)

| 대상 | 내용 |
|---|---|
| `compose.yaml` | **`shm_size: 1gb`** — 없으면 `CREATE INDEX`가 `No space left on device`로 죽는다. 디스크가 아니라 컨테이너 공유메모리 얘기라 메시지가 원인을 안 가리킨다 |
| `models/document.py` | 주석 속 정의가 설계와 불일치. 차원 확정 후 새로 작성 |
| `repositories/` | 지금 `SourceItem`(스키마)을 반환 중. **A안으로 감** — repository는 `SearchResult(document, score)` dataclass를 반환하고 service가 스키마로 변환 |
| `pyproject.toml` | `transformers<5` 고정 필요 (5.x에서 리랭커가 `prepare_for_model` 없음으로 죽음) |

### 참고 — 직전 시도의 실측 기록

Notion **`🧪 RAG 학습 메모 — 임베딩 · 인덱싱 · 리랭커`** (2026-08-12).
모델 선택은 이번에 다시 정하지만, **삽 포인트 목록은 모델과 무관하게 유효**하다:
배치 크기 8(100은 패딩 때문에 더 느림), 임베딩과 적재 분리 + 400건마다 커밋,
advisory lock으로 중복 적재 방지, HNSW opclass 불일치 시 **에러 없이** 완전탐색,
필터 걸면 후보가 조용히 반토막(`hnsw.iterative_scan`), `use_fp16=True`,
Gemini 모델명은 `client.models.list()`로 확인.
