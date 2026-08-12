# rag-study
## 강아지 AI 생활 비서 

```
AI-X 15기 4조 메인프로젝트에 활용할 기술들/환경세팅 등 학습하는 프로젝트
```

### Backend
- `uv`로 프로젝트 관리
- `python`버전 3.12 고정
    01. 3.14 전용 기능이 필요하지 않고 FastAPI·Qdrant·임베딩·PyTorch 등 여러 패키지를 엮는 48시간 작업이면 Windows와 CUDA까지 있어 최신 Python에서 발생할 수 있는 설치 변수를 줄이는 게 맞아서 3.12가 맞는거 같긴합니다.
    02. AI/ML 쪽은 Python 버전뿐만 아니라 numpy, scipy, torch, tensorflow 같은 라이브러리 버전끼리도 의존성이 있어서, 새 Python 버전을 쓰면 일부 패키지에서 아직 지원 버전이나 wheel이 안 맞는 경우가 있다고는 합니다. 
    - **결론** : 3.14 전용 기능이 필요 없고, Windows/CUDA 환경에서 torch·numpy 등 ML 패키지의 wheel 지원이 안정적인 **3.12로 고정**
- `fastapi` : `uv add fastapi --extra standard` = `uv add "fastapi[standard]"`
- `pydantic-settings` : env 관리 라이브러리
- llm : `gemini-3.1-flash-lite` api로 사용 예정 (로컬 llm은 테스트하기 어려운 pc환경)
- `google-genai` : gemini api 라이브러리

### Frontend
- `Next.js` (16.3.0)
- `React` (19.2.8)

### DB
- `pgvector` : 이미지 - pgvector/pgvector:pg18 (받아둠)

### data
- AI Hub — 59.반려견 성장 및 질병 관련 말뭉치 데이터
```
59.반려견 성장 및 질병 관련 말뭉치 데이터/3.개방데이터/1.데이터/
├── Training/
│   ├── 01.원천데이터/    TS_말뭉치데이터_{내과,안과,외과,치과,피부과}.zip     ~2.3MB
│   └── 02.라벨링데이터/  TL_질의응답데이터_{내과,안과,외과,치과,피부과}.zip  ~21.6MB
└── Validation/
```
> 원본 데이터는 용량/라이선스 문제로 저장소에 포함하지 않음. AI Hub에서 직접 내려받아 위 구조대로 `data/` 아래에 배치.

### 실행 방법
**Backend** (http://localhost:8000)
```bash
cd backend
uv sync              # .venv 생성 + 의존성 설치
uv run fastapi dev
```

**Frontend** (http://localhost:3000)
```bash
cd frontend
npm install
npm run dev
```

**DB**
```bash
docker run -d --name rag-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg18
```