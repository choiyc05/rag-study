import importlib
import pkgutil

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import controllers
from src.core.settings import get_settings

# 앱 기동 시점에 .env를 읽어 검증한다.
# 키가 없으면 여기서 바로 죽으므로, 요청이 들어온 뒤에야 발견하는 일이 없다.
settings = get_settings()

app = FastAPI(title="강아지 AI 생활 비서")

# CORS 설정 / .env에서 ORIGINS를 읽어온다. (배포 시점에만 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)


def register_routers(app: FastAPI) -> None:
    """src/controllers 아래의 모든 모듈에서 router를 찾아 자동 등록한다.

    자동화하는 것은 '등록'뿐이다. prefix와 tags는 각 컨트롤러의 APIRouter가
    직접 선언하므로, 파일명을 바꿔도 API 주소는 바뀌지 않는다.
    컨트롤러 파일을 추가하기만 하면 되고 이 파일은 고칠 일이 없다.
    """
    registered = 0
    for _, module_name, _ in pkgutil.iter_modules(controllers.__path__):
        if module_name.startswith("_"):
            continue

        module = importlib.import_module(f"src.controllers.{module_name}")
        router = getattr(module, "router", None)
        # 조용히 건너뛰면 오타 하나로 엔드포인트가 통째로 사라져도 서버는 멀쩡히 뜬다.
        # 404를 보고 나서야 알게 되므로 기동 시점에 막는다.
        if router is None:
            raise RuntimeError(
                f"src/controllers/{module_name}.py에 router가 없다. "
                "컨트롤러가 아니라면 파일명을 _로 시작하게 할 것."
            )

        app.include_router(router)
        registered += 1

    # controllers에 __init__.py가 없어(namespace package) 경로가 틀려도
    # 예외 대신 빈 목록이 돌아온다. 라우터 0개면 기동을 막는다.
    if registered == 0:
        raise RuntimeError("등록된 라우터가 없다. src/controllers 경로를 확인할 것.")


register_routers(app)


@app.get("/", tags=["test"])
def read_root():
    return {"Hello": "World"}
