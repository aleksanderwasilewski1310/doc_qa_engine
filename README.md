# Document QA + Loader API

![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS-S3-569A31?logo=amazons3&logoColor=white)
![PostgreSQL and pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)

This project is a FastAPI-based document ingestion and question answering system built around pgvector and AWS Bedrock.

The application now uses a single entry point in `api.py`, where the upload and QA routes are exposed together. The ingestion and retrieval logic is separated into focused modules under `app/`.

## Overview

The upload and question-answering workflow is:

1. A PDF is sent to `/upload` and saved to the S3 bucket at `uploads/<filename>`
2. The API writes a temporary local copy and sends it through the chunking pipeline
3. Each chunk is embedded with an AWS Bedrock embedding model and stored in PostgreSQL with pgvector
4. A user question is embedded and compared against stored vectors
5. The best matching chunks are gathered and passed as context to Claude through Bedrock
6. The final answer is returned to the client

## System architecture

```mermaid
flowchart LR
    Client[Client / Browser / curl] --> API[FastAPI app in api.py]
  API --> Upload["/upload"]
  API --> Ask["/api/v1/ask"]
  API --> AskForm["/api/v1/ask-form"]
  API --> Health["/health"]

    Upload --> S3["S3: uploads/<filename>"]
    Upload --> Chunking[app/chunking.py]
    Chunking --> DB[PostgreSQL + pgvector]

    Ask --> RAG[app/rag.py]
    RAG --> Embeddings[app/embeddings.py]
    RAG --> DB
    Embeddings --> Bedrock[AWS Bedrock embeddings]
    RAG --> LLM[Claude via ChatBedrock]
    LLM --> Response[Answer returned to client]
```

## Application structure

```text
.
├── api.py                     # single FastAPI entry point
├── app/
│   ├── __init__.py
│   ├── chunking.py            # PDF chunking and ingestion logic
│   ├── db_client.py           # PostgreSQL connection and inserts
│   ├── embeddings.py          # Bedrock embedding helper
│   ├── rag.py                 # retrieval + answer generation logic
│   └── main.py                # older project utility
├── docker/
│   ├── create_table.sql       # pgvector schema setup
│   ├── docker-compose.db.yml  # Postgres container
│   ├── docker-compose.loader.yml
│   ├── Dockerfile.loader
│   └── ...
├── infrastructure/
├── model_files/
├── src/
├── tests/
├── pyproject.toml
├── README.md
├── notes.txt
└── terraform.tfstate
```

## API endpoints

### GET /health

Returns service health status.

```bash
curl http://localhost:8000/health
```

### POST /upload

Uploads the original PDF to S3 at `s3://enterprise-document-storage-prod-eu-central-1/uploads/<filename>`, then processes a temporary local copy with the chunking pipeline and stores the resulting chunks in PostgreSQL. The response includes the S3 location.

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sample.pdf" \
  -F "distance=10"
```

Example response:

```json
{
  "status": "success",
  "filename": "sample.pdf",
  "distance": 10.0,
  "s3_location": "s3://enterprise-document-storage-prod-eu-central-1/uploads/sample.pdf"
}
```

### POST /api/v1/ask

Queries pgvector for the most relevant chunks and answers the question using the retrieved context.

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the key requirement in this document?",
    "top_k": 5
  }'
```

### POST /api/v1/ask-form

Same as the JSON route, but accepts form input.

```bash
curl -X POST "http://localhost:8000/api/v1/ask-form" \
  -F "question=What is the key requirement in this document?" \
  -F "top_k=5"
```

## RAG retrieval flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as API
    participant R as Retrieval Layer
    participant DB as PostgreSQL pgvector
    participant E as Embedding Model
    participant L as Bedrock LLM

    U->>A: POST /api/v1/ask
    A->>R: question + top_k
    R->>E: embed question
    E-->>R: question vector
    R->>DB: cosine similarity query
    DB-->>R: sorted relevant chunks
    R->>L: top_k chunks as context + question
    L-->>A: final answer
    A-->>U: answer + context + sources
```

## Database schema

The project stores chunk vectors in the `doc_chunks` table and uses pgvector similarity search.

```mermaid
erDiagram
    DOC_CHUNKS {
        int id PK
        varchar document_name
        timestamp timestamp
        text chunk_text
        float distance
        vector embedding
    }

    DOCUMENT_CHUNKS {
        int id PK
        timestamp created_at
        varchar file_name
        int chunk_index
        text content
        vector embedding
    }
```

  ### DynamoDB chat history

  Terraform defines the `doc_qa_chat_history` table in `infrastructure/dynamodb.tf` for chat-session messages. The table uses `session_id` as its partition key and `message_id` as its sort key, allowing multiple messages per session. It uses on-demand billing (`PAY_PER_REQUEST`) and enables TTL on the `ttl` attribute; write `ttl` as a Unix timestamp in seconds when a message should expire. The Terraform provider is configured for `eu-central-1`, and the table is tagged with the `poc` environment and `doc_qa_engine` project.

  To review and apply the Terraform configuration:

  ```bash
  cd infrastructure
  terraform init
  terraform plan
  terraform apply
  ```

  Review the plan before applying: Terraform operates on all resources configured in the `infrastructure/` directory, not only the DynamoDB table. This table is provisioned independently; the current document retrieval workflow continues to store chunks in PostgreSQL with pgvector.

## Storage and query behavior

The active retrieval table is `doc_chunks`.

Each row contains:

- `id` — primary key
- `document_name` — source file name
- `timestamp` — ingestion time
- `chunk_text` — chunk content
- `distance` — optional metadata field
- `embedding` — vector representation from Bedrock

The retrieval process does the following:

- embeds the incoming question
- runs a vector similarity search against the stored embeddings
- orders results by distance or cosine similarity
- sends the highest-ranked chunks as contextual input to the LLM
- keeps the relevant context ordered from strongest match to weaker match

## Chunking behavior

The project uses the chunking pipeline in `app/chunking.py` to split PDFs into semantically manageable blocks.

```mermaid
flowchart TD
    PDF[Uploaded PDF] --> Load[Load document text]
    Load --> Split[RecursiveCharacterTextSplitter]
    Split --> Metadata[Add file_name / chunk metadata]
    Metadata --> Embed[Generate embeddings]
    Embed --> Store[Insert into PostgreSQL]
```

## Local development

### Requirements

- Python 3.13+
- Docker and Docker Compose
- PostgreSQL with pgvector enabled
- AWS Bedrock access and valid credentials

### Install dependencies

```bash
pip install -e .
```

Or with uv:

```bash
uv sync
```

### Start the database

```bash
cd docker
docker compose -f docker-compose.db.yml up -d
```

### Start the API

From the project root:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`

## Environment variables

### S3 document storage

The upload endpoint currently uses the bucket `enterprise-document-storage-prod-eu-central-1` in `eu-central-1`; these values are configured in `api.py`, not read from environment variables. Provision the bucket with the Terraform configuration in `infrastructure/s3.tf` before using uploads. The configured bucket blocks public access, enables KMS server-side encryption, and has versioning enabled.

The AWS identity used by the API must be allowed to write objects under the `uploads/` prefix. For example, grant `s3:PutObject` on `arn:aws:s3:::enterprise-document-storage-prod-eu-central-1/uploads/*`. Configure credentials through the standard AWS SDK credential chain (environment variables, an AWS profile, or the workload's IAM role).

### PostgreSQL

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres
```

### AWS / Bedrock

```bash
AWS_REGION=eu-central-1
AWS_DEFAULT_REGION=eu-central-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
```

## Docker configuration

The loader container is configured to run the unified app entry point.

```yaml
services:
  loader-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.loader
    ports:
      - "8000:8000"
    command: uvicorn api:app --host 0.0.0.0 --port 8000
```

From the project root, start the API with Docker Compose:

```powershell
docker compose -f docker/docker-compose.loader.yml up --build
```

Open `http://localhost:8000/docs` for the interactive API documentation, or `http://localhost:8000/health` to check that the API is running. To run it in the background and follow its logs:

```powershell
docker compose -f docker/docker-compose.loader.yml up --build -d
docker compose -f docker/docker-compose.loader.yml logs -f
```

Stop the container with:

```powershell
docker compose -f docker/docker-compose.loader.yml down
```

The Compose file starts the API only. Endpoints that use AWS or PostgreSQL also require those services to be reachable and credentials/configuration to be available inside the container.

## Notes

- The old split entry-point files were removed in favor of the single app module `api.py`.
- Business logic is organized into individual modules for easier reuse and maintenance.
- Retrieval is grounded in stored vector data and is designed to keep the most relevant context near the start of the prompt.

## Summary

This repository combines PDF ingestion, chunking, embedding generation, pgvector storage, and Bedrock-powered RAG to create a lightweight document QA system with a clean single-app FastAPI interface.
