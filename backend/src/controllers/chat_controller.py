"""Controller = HTTP 진입점. "어떤 URL로 들어와서 어떤 형태로 나가는가"만 안다.

여기서 하는 일은 딱 세 가지다.
    1. 경로 / 메서드 / 상태코드 정의
    2. 요청을 schema로 받기 (검증은 FastAPI가 대신 해줌)
    3. service 호출 후 결과 반환

계산도, DB 접근도 하지 않는다. 이 파일이 길어지고 있다면 service로 옮길 로직이 섞인 것이다.

URL prefix와 /docs 태그는 아래 APIRouter가 직접 소유한다.
main.py는 이 파일을 찾아 등록만 할 뿐 경로에 관여하지 않으므로,
이 API의 주소를 알고 싶으면 여기만 보면 된다.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from src.repositories.document_repository import DocumentRepository
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_service() -> ChatService:
    """의존성 조립. 나중에 DB 세션이 생기면 여기서 함께 주입한다."""
    return ChatService(repository=DocumentRepository())


@router.post("", response_model=ChatResponse)
def ask(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    return service.ask(request)
