-- =====================================================================
-- 01_schema.sql
-- 생활비서 RAG - documents 테이블 스키마
-- 실행 순서: 01_schema.sql -> 02_trigger.sql -> (데이터 적재) -> indexes.sql
-- =====================================================================

-- pgvector 확장 활성화 (VECTOR 타입 사용을 위해 필수)
CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------
-- documents : RAG 검색 대상 문서 청크
-- ---------------------------------------------------------------------
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(1024),

    category VARCHAR(50) NOT NULL
        CHECK (category IN ('policy','food','care','travel','emergency')),
    subcategory VARCHAR(50) NOT NULL,

    source VARCHAR(100),
    source_type VARCHAR(50)
        CHECK (source_type IN ('pdf','web','api','manual')),
    source_url TEXT,
    document_title TEXT,
    section TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ---------------------------------------------------------------------
-- 컬럼 설명 (\d+ documents 또는 DBeaver/pgAdmin에서 확인 가능)
-- ---------------------------------------------------------------------
COMMENT ON TABLE  documents                 IS '생활비서 RAG 문서 청크';

COMMENT ON COLUMN documents.id              IS 'UUID로 생성되는 문서/Chunk 고유 ID';
COMMENT ON COLUMN documents.content         IS '실제 RAG 검색 대상이 되는 Chunk';
COMMENT ON COLUMN documents.embedding       IS '1024차원 벡터';
COMMENT ON COLUMN documents.category        IS '대분류';
COMMENT ON COLUMN documents.subcategory     IS '세부 분류';
COMMENT ON COLUMN documents.source          IS '자료 출처';
COMMENT ON COLUMN documents.source_type     IS '자료 유형';
COMMENT ON COLUMN documents.source_url      IS '원문 URL/출처 링크';
COMMENT ON COLUMN documents.document_title  IS '원본 문서 제목';
COMMENT ON COLUMN documents.section         IS '원본 문서의 장/절';
COMMENT ON COLUMN documents.metadata        IS '메타 데이터/추가적인 부가 정보';
COMMENT ON COLUMN documents.created_at      IS '생성/저장 시각';
COMMENT ON COLUMN documents.updated_at      IS '수정 시각';
