"""Model = DB 테이블 그 자체. 컬럼, 타입, 인덱스, 제약조건을 정의한다.

아직 sqlalchemy / pgvector를 설치하지 않아 실제 정의는 주석으로만 둔다.
설치 후(uv add sqlalchemy pgvector) 아래를 살리면 된다.

    from pgvector.sqlalchemy import Vector
    from sqlalchemy import Index, Integer, String, Text
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


    class Base(DeclarativeBase):
        pass


    class Document(Base):
        __tablename__ = "documents"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        content: Mapped[str] = mapped_column(Text, nullable=False)
        category: Mapped[str] = mapped_column(String(20))       # 내과/안과/외과/치과/피부과
        embedding: Mapped[list[float]] = mapped_column(Vector(768))

        __table_args__ = (
            Index(
                "ix_documents_embedding",
                "embedding",
                postgresql_using="hnsw",
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
        )

Schema(ChatResponse)와 비교해보면 차이가 명확하다.
여기에는 embedding 벡터 768개가 들어있지만, 프론트에 그걸 내려줄 일은 없다.
"""
