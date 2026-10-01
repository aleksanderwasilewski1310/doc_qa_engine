import base64
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import boto3
from botocore.exceptions import BotoCoreError
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

load_dotenv()  # Load environment variables from .env file
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Multimodal Document QA Engine",
    description="Isolated backend service handling multimodal document processing and LLM synthesis via private AWS infrastructure.",
)

# Initialize AWS SDK clients.
# In production, these rely on IAM Roles assigned to the container execution environment (e.g., ECS Task Role).
region = os.getenv("AWS_REGION", "eu-central-1")

# Client for invoking low-latency SageMaker endpoints running custom inference containers inside the VPC
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=region)

# Client for calling foundation models (e.g., Claude 3.5 Sonnet) via managed API (AWS Bedrock) over PrivateLink
bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

# Client for encrypted object storage operations
s3_client = boto3.client("s3", region_name=region)

# Environment variables populated via ECS task definition
SAGEMAKER_VISION_ENDPOINT = os.getenv(
    "SAGEMAKER_VISION_ENDPOINT", "vision-ocr-endpoint"
)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "enterprise-document-storage-dev")


def invoke_sagemaker_vision(image_bytes: bytes) -> str:
    """
    Invokes the custom vision/OCR model deployed to the SageMaker endpoint.
    The deployed Hugging Face container expects a JSON payload containing a base64-encoded image.
    """
    try:
        payload = {"inputs": base64.b64encode(image_bytes).decode("utf-8")}

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_VISION_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload).encode("utf-8"),
        )

        raw_response = response["Body"].read().decode("utf-8")
        result = json.loads(raw_response)

        if isinstance(result, dict):
            return result.get(
                "extracted_text", result.get("generated_text", str(result))
            )

        if isinstance(result, list) and result:
            first_item = result[0]
            if isinstance(first_item, dict):
                return first_item.get(
                    "extracted_text", first_item.get("generated_text", str(first_item))
                )
            return str(first_item)

        return str(result)

    except (AttributeError, BotoCoreError, KeyError, TypeError, ValueError) as e:
        detail = f"SageMaker Inference Failure: {e!s}"
        if "Connection reset by peer" in str(e) or "ModelError" in str(e):
            detail = "SageMaker Inference Failure: the vision endpoint crashed or failed to process the request. Please check the SageMaker endpoint logs."
        raise HTTPException(status_code=500, detail=detail)


def generate_llm_response(prompt: str, context: str) -> str:
    """
    Executes a contextual generation call against AWS Bedrock (Claude 3.5 Sonnet).

    Architectural Note:
    Uses strict system prompts to constrain LLM scope to the extracted context (mitigating hallucination risks).
    """
    formatted_prompt = f"""System: You are an enterprise technical documentation assistant.
Answer EXCLUSIVELY using the context provided below from the ingested document/image.
If the information is not present in the context, explicitly state that you cannot answer based on the provided data.

Document Context:
{context}

User Inquiry: {prompt}
Response:"""

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": formatted_prompt}],
        "temperature": 0.1,  # Low temperature to prioritize deterministic extraction over creativity
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        response_body = json.loads(response.get("body").read())
        return response_body["content"][0]["text"]
    except (AttributeError, BotoCoreError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(
            status_code=500, detail=f"Bedrock Runtime Execution Error: {e!s}"
        )


@app.post("/api/v1/query-document")
async def process_document_query(
    question: Annotated[str, Form()], file: Annotated[UploadFile, File()]
):
    """
    Ingest payload endpoint:
    1. Reads multi-part binary stream.
    2. Persists asset to S3 with customer-managed KMS encryption.
    3. Invokes SageMaker OCR for optical layout parsing.
    4. Passes extracted text to Bedrock LLM via RAG synthesis loop.
    """
    file_bytes = await file.read()

    # Secure object storage write using Server-Side Encryption (SSE-KMS)
    s3_key = f"uploads/{file.filename}"
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
        Body=file_bytes,
        ServerSideEncryption="aws:kms",
    )

    # Step 1: Extract structural text & tables via dedicated SageMaker OCR endpoint
    extracted_text = invoke_sagemaker_vision(file_bytes)

    # Step 2: Contextual QA synthesis using AWS Bedrock foundation model
    answer = generate_llm_response(prompt=question, context=extracted_text)

    return {
        "status": "success",
        "s3_location": f"s3://{S3_BUCKET_NAME}/{s3_key}",
        "extracted_context_length": len(extracted_text),
        "answer": answer,
    }


@app.post("/api/v1/upload-and-chunk")
async def upload_and_chunk(
    file: Annotated[UploadFile, File()],
    race_distance: Annotated[float, Form()],
):
    """Upload a PDF and run the chunking pipeline (`chunking.main`).

    The uploaded file is written to a temporary file and `chunking.main`
    is executed in a threadpool to avoid blocking the event loop.
    """
    # Save uploaded file to a temporary directory
    tempdir = tempfile.mkdtemp()
    try:
        tmp_path = Path(tempdir) / file.filename
        contents = await file.read()
        tmp_path.write_bytes(contents)

        # Import here to avoid circular imports at module load time
        try:
            from . import chunking
        except ImportError:
            import chunking

        # Run the synchronous chunking.main in a threadpool
        try:
            await run_in_threadpool(chunking.main, str(tmp_path), float(race_distance))
        except Exception as err:
            logger.exception("Chunking failed")
            raise HTTPException(
                status_code=500, detail=f"Chunking failed: {err}"
            ) from err

        return {"status": "success", "filename": file.filename}
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(tempdir)
        except OSError:
            pass
