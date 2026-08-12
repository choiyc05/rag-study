"""Service = 비즈니스 로직. "무엇을 어떤 순서로 하는가"를 안다.

RAG 파이프라인의 절차(질문 임베딩 → 유사 문서 검색 → 프롬프트 조립 → LLM 호출)가
전부 이 계층에 있다.

중요: 이 파일에는 HTTP 관련 코드가 하나도 없다.
Request, HTTPException, status_code 같은 걸 쓰지 않는다.
그래야 나중에 CLI나 배치 작업에서도 이 서비스를 그대로 재사용할 수 있다.
"""

from src.repositories.document_repository import DocumentRepository
from src.schemas.chat import ChatRequest, ChatResponse


class ChatService:
    def __init__(self, repository: DocumentRepository) -> None:
        # 직접 만들지 않고 주입받는다 → 테스트할 때 가짜 repository로 교체 가능
        self.repository = repository

    def ask(self, request: ChatRequest) -> ChatResponse:
        # 1) 질문을 임베딩으로 변환
        embedding = self._embed(request.question)

        # 2) 유사 문서 검색 (DB 접근은 repository에 위임)
        sources = self.repository.search_similar(embedding, request.top_k)

        # 3) 검색 결과를 컨텍스트로 붙여 LLM 호출
        answer = self._generate(request.question, sources)

        return ChatResponse(answer=answer, sources=sources)

    def _embed(self, text: str) -> list[float]:
        # TODO: google-genai 임베딩 모델로 교체
        return [0.0] * 768

    def _generate(self, question: str, sources: list) -> str:
        # TODO: settings.GEMINI_API_KEY로 genai 클라이언트 만들어 호출
        context = "\n".join(s.content for s in sources)
        return f"[임시 응답] 질문: {question} / 참고 문서 {len(sources)}건\n{context}"
