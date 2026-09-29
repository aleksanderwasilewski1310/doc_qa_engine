from __future__ import annotations

import boto3
from typing import Any

import uvicorn
import shutil
import tempfile
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from app.rag import answer_question
from app.chunking import main as chunk_main
from app.ragas_eval import evaluate_rag

S3_BUCKET_NAME = "enterprise-document-storage-prod-eu-central-1"
s3_client = boto3.client("s3", region_name="eu-central-1")

app = FastAPI(
    title="Document QA + Loader API",
    description="Single FastAPI app that supports PDF upload/chunking and pgvector RAG question answering.",
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_and_chunk(
    file: UploadFile = File(...),
    distance: float = Form(...),
):
    """Upload PDF, chunk it, and store chunks in the database."""


    if file.filename is None:
        raise ValueError("No filename provided for upload.")

    tmpdir = tempfile.mkdtemp(prefix="loader_api_")
    target_path = Path(tmpdir) / file.filename

    try:
        contents = await file.read()
        s3_key = f"uploads/{file.filename}"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=contents)
        target_path.write_bytes(contents)
        await run_in_threadpool(chunk_main, str(target_path), float(distance))
        return {
            "status": "success",
            "filename": file.filename,
            "distance": float(distance),
            "s3_location": f"s3://{S3_BUCKET_NAME}/{s3_key}",
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/api/v1/ask")
async def ask_question(payload: AskRequest) -> dict[str, Any]:
    return answer_question(payload.question, top_k=payload.top_k)


@app.post("/api/v1/ask-form")
async def ask_question_form(question: str, top_k: int = 5) -> dict[str, Any]:
    return answer_question(question, top_k=top_k)


@app.post("/api/v1/evaluate")
async def evaluate_rag_endpoint(payload: EvaluationRequest) -> dict[str, Any]:
    return evaluate_rag(
        question=payload.question,
        answer=payload.answer,
        contexts=payload.contexts,
        ground_truth=payload.ground_truth,
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
