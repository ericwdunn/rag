-- One-time setup: createdb rag && psql -d rag -f schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    source      text        NOT NULL,   -- e.g. docs/handbook.md
    chunk_index int         NOT NULL,   -- paragraph position in that file
    content     text        NOT NULL,
    embedding   vector(768) NOT NULL,   -- nomic-embed-text returns 768 dims
    UNIQUE (source, chunk_index)
);

-- Approximate nearest-neighbor index for cosine distance (<=>).
-- At a few dozen rows the planner ignores it and scans; it pays off at thousands.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
