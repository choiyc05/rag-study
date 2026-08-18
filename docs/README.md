# docs

`rag-study` 프로젝트 진행 기록. 세션이 끊겨도 여기만 읽으면 맥락을 복구할 수 있도록 정리한다.

| 문서 | 내용 |
|---|---|
| [setup.md](setup.md) | 환경 세팅 (Python 3.12, uv, 백엔드 구조, Docker, git 설정) |
| [dataset-analysis.md](dataset-analysis.md) | AI Hub 데이터셋 실측 분석 결과 |
| [rag-design.md](rag-design.md) | RAG 설계 결정과 그 근거 |
| **[roadmap.md](roadmap.md)** | **단계별 실행 계획. 다음에 뭘 할지는 여기부터** |
| **[experiments.md](experiments.md)** | **실험 기록. 이 프로젝트의 결과물** |
| [data-expansion.md](data-expansion.md) | 데이터 확장 전략 (수집 · 편입 방식) |
| [mentor-feedback.md](mentor-feedback.md) | 멘토 피드백 원문 기록 + 반영 현황 |

## 현재 상태 (2026-08-17 기준)

- 코드는 **2026-08-13 이후 변경 없음.** 마지막 커밋 `4e810d0`은 문서 3개 수정
- 멘토 피드백 반영해 **실행 계획 수립 완료** → [roadmap.md](roadmap.md)
- **RAG 구현은 아직 0%.** 다음 착수 지점은 roadmap의 **Phase 0**

### 2026-08-17 — 멘토 피드백 반영

다른 레포에서 빠르게 만든 RAG 프로토타입으로 피드백을 받았다.
**그 레포의 코드는 가져오지 않는다** (빠르게 만드느라 이해하지 못한 채 넘어간 부분이
많아, 없는 것으로 치고 계획부터 다시 세운다).

피드백 원문은 [mentor-feedback.md](mentor-feedback.md)에 그대로 남겨두었다 —
나중에 "이건 내 판단인가 멘토 조언인가"를 구분할 수 있어야 하기 때문이다.

기존 설계에 **이미 반영돼 있던 것**: EDA 선행, 리랭킹, Hit Rate/MRR(=지표 1),
LLM-as-a-judge(=지표 2), 리더보드 맹신 금지, 청킹 전략 비교.

**새로 들어온 것** (대조표는 [roadmap.md](roadmap.md) §1):

| 항목 | 반영 위치 |
|---|---|
| 하이브리드 검색 (BM25 + RRF) | roadmap §3 — 한국어 BM25를 앱단/DB단 중 어디서 돌릴지 결정 필요 |
| Parent-Child 청킹 | roadmap §4 — 기존 `content`/`payload` 구조로 스키마 변경 없이 가능 |
| Semantic chunking | 청킹 비교군에 추가 |
| 주기적 크롤링 · 유튜브 STT · Docling | [data-expansion.md](data-expansion.md) |
| 실험 수치 기록 양식 | [experiments.md](experiments.md) 신설 |
| 상충 데이터를 지우지 말 것 | data-expansion §9 |

**용어 정리:** 멘토가 말한 `Hit Rate@k`와 기존 문서의 `Recall@k`는
정답이 1개인 known-item 설계에서 **같은 값**이다. `Hit Rate@k`로 통일한다.

### 데이터 확장에 대한 판단 (신규)

AI Hub QA는 **질문을 임베딩**하는데 앞으로 모을 데이터는 대부분 **문서**라,
그냥 넣으면 대칭/비대칭 매칭이 한 인덱스에 섞인다.
→ **doc2query**(문서에서 질문을 생성해 그 질문을 인덱싱)로 형태를 맞춘다.

그리고 지금 부족한 건 양이 아니라 **주제**다. AI Hub는 5개 진료과의 *질병* 상담뿐이고
사료·행동·훈련·미용은 **한 건도 없다** — 서비스 이름은 "생활 비서"인데.
→ 수집 전에 **커버리지 공백 진단**부터. 상세는 [data-expansion.md](data-expansion.md).

## 이전 기록 (2026-08-13 기준)

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

**→ [roadmap.md](roadmap.md)의 Phase 0.** 단계별 상세와 완료 기준은 전부 거기 있다.
여기서는 중복해서 적지 않는다 (두 곳에 적으면 반드시 어긋난다).

요약하면 — `prepare_data.py`(zip → JSONL) → 구어체 평가셋 → 모델 3개 임베딩(RunPod)
→ numpy로 Hit Rate/MRR 비교 → **모델·차원 확정.** 여기까지 DB는 필요 없다.

### 실행 환경 (2026-08-17 확정) — 막혀 있던 항목 해소

- **로컬 GPU: RTX 3050 6GB.** fp16이면 임베딩+리랭커 동시 로드가 ~2.2GB라 **여유 있다.**
  기존 문서의 `~4GB`는 fp32 기준이었다
- **오프라인 임베딩: RunPod** (연습 목적). 이 규모(21,606건)는 로컬로도 모델 3개
  ~1시간이라 가성비 얘기가 아니고, 비용도 $1 미만이다.
  원격 GPU가 실제로 필요해지는 건 **Phase 3의 유튜브 STT**
- ⚠️ **새 걱정거리는 VRAM이 아니라 리랭커 지연.** 3050에서 20쌍 리랭킹이 2초 안팎으로
  추정된다 → Phase 1 R-2에서 **실측이 최우선**

상세는 [rag-design.md](rag-design.md) "실행 환경".

### 계류 중인 수정

하기로 정했으나 아직 손 안 댄 것들. **[roadmap.md](roadmap.md) Phase 2에서 함께 처리**한다
(`compose.yaml` `shm_size`, `models/document.py` 재작성, repository 반환 타입,
`transformers<5` 고정).

### 참고 — 직전 시도의 실측 기록

Notion **`🧪 RAG 학습 메모 — 임베딩 · 인덱싱 · 리랭커`** (2026-08-12).
모델 선택은 이번에 다시 정하지만 **삽 포인트 목록은 모델과 무관하게 유효**하며,
[experiments.md](experiments.md)의 "삽질 기록"에 옮겨두었다.
