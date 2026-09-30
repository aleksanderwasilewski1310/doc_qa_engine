"""Bedrock embedding helper utilities.

Reads AWS credentials from environment (optionally via .env) and provides
`embed_texts` to obtain embeddings from Bedrock models like Amazon Titan
or Cohere Embed.
"""

from __future__ import annotations

import json
import logging
import os
import tiktoken
from typing import List, Optional

from dotenv import load_dotenv
import boto3

load_dotenv()
logger = logging.getLogger(__name__)


def _get_bedrock_client():
    region = (
        os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "eu-central-1"
    )

    # Initialize boto3 client
    kwargs = {"region_name": region}
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
        kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")

    return boto3.client("bedrock-runtime", **kwargs)


def _embed_single_text_titan(text: str, client, model_id: str) -> List[float]:
    """Generates embedding for a single string using Amazon Titan Embeddings V2."""
    payload = {
        "inputText": text,
        "dimensions": 1024,  # Available options: 256, 512, 1024
        "normalize": True,
    }

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )

    response_body = json.loads(response.get("body").read())
    return response_body["embedding"]


def _embed_cohere_multilingual(
    texts: List[str], client, model_id: str
) -> List[List[float]]:
    """Cohere v3 supports batching multiple texts natively in a single API call."""
    payload = {"texts": texts, "input_type": "search_document", "truncate": "END"}

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )

    response_body = json.loads(response.get("body").read())
    return response_body["embeddings"]


def embed_texts(texts: List[str], model_id: Optional[str] = None) -> List[List[float]]:
    """Return embeddings for the provided list of texts using AWS Bedrock.

    Args:
        texts: List of input strings to embed.
        model_id: Optional Bedrock model identifier. Defaults to
                  environment variable `AWS_EMBEDDING_MODEL` or
                  `amazon.titan-embed-text-v2:0`.

    Returns:
        A list of embedding vectors (list of floats) matching the order of `texts`.
    """
    if not texts:
        return []

    model = (
        model_id or os.getenv("AWS_EMBEDDING_MODEL") or "amazon.titan-embed-text-v2:0"
    )
    client = _get_bedrock_client()

    # Estimate token usage before making requests. Prefer `tiktoken` when available.
    def _count_tokens(text: str) -> int:
        if tiktoken is not None:
            try:
                # Attempt to choose encoding for the model; fall back to cl100k_base
                try:
                    enc = tiktoken.encoding_for_model(model)
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except Exception:
                pass
        # Fallback heuristic: approximate tokens as chars/4
        return max(1, int(len(text) / 4))

    token_counts = [_count_tokens(t) for t in texts]
    total_tokens = sum(token_counts)
    logger.info(
        "Embedding request: model=%s texts=%d total_tokens=%d",
        model,
        len(texts),
        total_tokens,
    )

    try:
        # Cohere supports passing the entire batch of texts natively
        if "cohere" in model.lower():
            return _embed_cohere_multilingual(texts, client, model)

        # Titan V2 requires iterating over each text string individually
        embeddings = []
        for text in texts:
            vector = _embed_single_text_titan(text, client, model)
            embeddings.append(vector)

        logger.info("Embedding response received: model=%s texts=%d", model, len(texts))
        return embeddings

    except Exception as e:
        logger.exception("Bedrock embedding invocation failed: %s", e)
        raise


__all__ = ["embed_texts"]
