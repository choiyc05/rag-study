"""Repository = DB 접근 전담. "어떻게 가져오는가"만 안다.

비즈니스 규칙은 몰라야 한다. 여기서 "질문이 비어있으면 에러" 같은 판단을 하면 안 되고,
"이 벡터와 가까운 문서 N개 줘" 같은 조회 요청만 처리한다.

DB 연결 전이라 지금은 더미 데이터를 돌려준다.
"""

from src.schemas.chat import SourceItem


class DocumentRepository:
    def search_similar(self, embedding: list[float], top_k: int) -> list[SourceItem]:
        """임베딩과 코사인 거리가 가까운 문서 top_k개 조회.

        DB 연결 후에는 이 자리에 아래 같은 쿼리가 들어간다.
            SELECT content, 1 - (embedding <=> :embedding) AS score
            FROM documents
            ORDER BY embedding <=> :embedding
            LIMIT :top_k
        """
        # TODO: pgvector 연결 후 실제 쿼리로 교체
        dummy = [
            SourceItem(content="강아지 슬개골 탈구는 소형견에서 흔하게 나타납니다.", score=0.92),
            SourceItem(content="예방을 위해 미끄럽지 않은 바닥재를 사용하는 것이 좋습니다.", score=0.87),
        ]
        return dummy[:top_k]
