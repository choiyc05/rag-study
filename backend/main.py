from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.controllers import chat_controller
from src.core.settings import get_settings

# 앱 기동 시점에 .env를 읽어 검증한다.
# 키가 없으면 여기서 바로 죽으므로, 요청이 들어온 뒤에야 발견하는 일이 없다.
settings = get_settings()

app = FastAPI(title="강아지 AI 생활 비서")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

# controller(라우터) 등록. 기능이 늘어나면 이 줄만 추가된다.
app.include_router(chat_controller.router)

@app.get("/", tags=["test"])
def read_root():
    return {"Hello": "World"}
