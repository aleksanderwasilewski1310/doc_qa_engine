"""FastAPI entry point for document upload, question answering, and evaluation."""

import math
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any  # pylint: disable=no-name-in-module

import boto3
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.chunking import main as chunk_main
from app.rag import answer_question
from app.ragas_eval import evaluate_rag

S3_BUCKET_NAME = "enterprise-document-storage-prod-eu-central-1"
S3_CLIENT = boto3.client("s3", region_name="eu-central-1")

app = FastAPI(  # pylint: disable=invalid-name
    title="Document QA + Loader API",
    description=(
        "Single FastAPI app that supports PDF upload/chunking and pgvector "
        "RAG question answering."
    ),
)


def _safe_uploaded_filename(filename: str | None) -> str:
    """Validate a client-supplied name and return its safe PDF basename."""
    if filename is None or not filename.strip():
        raise HTTPException(status_code=400, detail="A filename is required.")

    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    return safe_name


def _validated_distance(distance: float) -> float:
    """Reject non-finite and non-positive chunking distances."""
    if not math.isfinite(distance) or distance <= 0:
        raise HTTPException(
            status_code=400,
            detail="Distance must be a positive finite number.",
        )
    return distance


class AskRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """Question and retrieval limit for the JSON QA endpoint."""

    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, gt=0)


class EvaluationRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """Inputs used to evaluate a generated answer against supplied contexts."""

    question: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(..., min_length=1, max_length=40000)
    contexts: list[str] = Field(..., min_length=1)
    ground_truth: str | None = Field(default=None, max_length=40000)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Report that the API process is available."""
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
        S3_CLIENT.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=contents)
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
    """Answer a JSON question using retrieved document chunks."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return answer_question(question, top_k=payload.top_k)


@app.post("/api/v1/ask-form")
async def ask_question_form(question: str, top_k: int = 5) -> dict[str, Any]:
    """Answer a question submitted as form fields."""
    normalized_question = question.strip()
    if not normalized_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")
    return answer_question(normalized_question, top_k=top_k)


@app.post("/api/v1/evaluate")
async def evaluate_rag_endpoint(payload: EvaluationRequest) -> dict[str, Any]:
    """Evaluate an answer against its question and supporting contexts."""
    if not payload.contexts or any(not context.strip() for context in payload.contexts):
        raise HTTPException(
            status_code=400,
            detail="At least one non-empty context is required.",
        )
    return evaluate_rag(
        question=payload.question,
        answer=payload.answer,
        contexts=payload.contexts,
        ground_truth=payload.ground_truth,
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
