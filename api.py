from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

import boto3
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.chunking import main as chunk_main
from app.rag import answer_question
from app.ragas_eval import evaluate_rag

S3_BUCKET_NAME = "enterprise-document-storage-prod-eu-central-1"
s3_client = boto3.client("s3", region_name="eu-central-1")

app = FastAPI(
    title="Document QA + Loader API",
    description="Single FastAPI app that supports PDF upload/chunking and pgvector RAG question answering.",
)


def _safe_uploaded_filename(filename: str | None) -> str:
    if filename is None or not filename.strip():
        raise HTTPException(status_code=400, detail="A filename is required.")

    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    return safe_name


def _validated_distance(distance: float) -> float:
    if not distance or not float(distance).is_finite() or float(distance) <= 0:
        raise HTTPException(
            status_code=400,
            detail="Distance must be a positive finite number.",
        )
    return float(distance)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, gt=0)


class EvaluationRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(..., min_length=1, max_length=40000)
    contexts: list[str] = Field(..., min_length=1)
    ground_truth: str | None = Field(default=None, max_length=40000)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_and_chunk(
    file: Annotated[UploadFile, File()],
    distance: Annotated[float, Form()],
):
    """Upload PDF, chunk it, and store chunks in the database."""

    safe_name = _safe_uploaded_filename(file.filename)
    validated_distance = _validated_distance(float(distance))

    tmpdir = tempfile.mkdtemp(prefix="loader_api_")
    target_path = Path(tmpdir) / safe_name

    try:
        contents = await file.read()
        s3_key = f"uploads/{safe_name}"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=contents)
        target_path.write_bytes(contents)
        await run_in_threadpool(chunk_main, str(target_path), validated_distance)
        return {
            "status": "success",
            "filename": safe_name,
            "distance": validated_distance,
            "s3_location": f"s3://{S3_BUCKET_NAME}/{s3_key}",
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/api/v1/ask")
async def ask_question(payload: AskRequest) -> dict[str, Any]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return answer_question(question, top_k=payload.top_k)


@app.post("/api/v1/ask-form")
async def ask_question_form(question: str, top_k: int = 5) -> dict[str, Any]:
    normalized_question = question.strip()
    if not normalized_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")
    return answer_question(normalized_question, top_k=top_k)


@app.post("/api/v1/evaluate")
async def evaluate_rag_endpoint(payload: EvaluationRequest) -> dict[str, Any]:
    if not payload.contexts or any(not context.strip() for context in payload.contexts):
        raise HTTPException(status_code=400, detail="At least one non-empty context is required.")
    return evaluate_rag(
        question=payload.question,
        answer=payload.answer,
        contexts=payload.contexts,
        ground_truth=payload.ground_truth,
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
