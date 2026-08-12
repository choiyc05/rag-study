"""Schema = 외부(프론트)와 주고받는 데이터의 모양. DTO 역할.

DB 테이블 구조와 일부러 분리한다.
- 요청에는 있지만 DB에 저장 안 하는 값이 있고 (예: top_k)
- DB에는 있지만 프론트에 절대 노출하면 안 되는 값이 있다 (예: 비밀번호, 내부 점수)
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat 요청 바디. FastAPI가 이 타입으로 자동 검증해준다."""

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class SourceItem(BaseModel):
    """답변 근거로 사용한 문서 조각 (RAG의 출처 표시용)."""

    content: str
    score: float


class ChatResponse(BaseModel):
    """POST /chat 응답 바디."""

    answer: str
    sources: list[SourceItem] = []
