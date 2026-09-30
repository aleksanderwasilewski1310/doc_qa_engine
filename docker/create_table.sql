-- STEP 1: Activate the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- STEP 2: Cleanup (optional)
DROP TABLE IF EXISTS document_chunks;

-- STEP 3: Table for PDF document chunks
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Document and chunk data
    file_name VARCHAR(255),               -- PDF file name (e.g. 'umowa.pdf')
    chunk_index INT,                      -- Order of the chunk in the document
    content TEXT NOT NULL,                -- Text of the extracted chunk

    -- Embedding vector (dimension 1024 for Titan V2 embeddings)
    embedding vector(1024)
);

-- STEP 4: HNSW index to speed up semantic search (Cosine Distance)
CREATE INDEX idx_document_chunks_embedding
ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- STEP 5: Additional table for generic document chunks
CREATE TABLE IF NOT EXISTS doc_chunks (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    chunk_text TEXT NOT NULL,
    distance REAL,
    embedding vector(1024)
);
