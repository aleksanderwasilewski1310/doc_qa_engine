import os
from typing import Any

from fastapi import HTTPException
from langchain_aws.chat_models import ChatBedrock

from app.db_client import get_connection
from app.embeddings import embed_texts


def build_bedrock_llm():
    import boto3

    region = os.getenv("AWS_REGION", "eu-central-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    return ChatBedrock(
        model_id="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        client=client,
        model_kwargs={"temperature": 0.1},
    )


def retrieve_relevant_chunks(question: str, top_k: int = 5) -> dict[str, Any]:
    """Query pgvector for the most similar chunks and return them as retrieval candidates."""
    query_vector = embed_texts([question])[0]
    embedding_literal = (
        "[" + ",".join(str(float(value)) for value in query_vector) + "]"
    )

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
                SELECT id, document_name, chunk_text,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM doc_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
            (embedding_literal, embedding_literal, top_k),
        )
        rows = cur.fetchall()

    if not rows:
        return {"context": "", "sources": [], "chunks": []}

    chunks: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    context_parts: list[str] = []
    for chunk_id, document_name, chunk_text, similarity in rows:
        similarity_score = float(similarity) if similarity is not None else 0.0
        chunk_record = {
            "id": chunk_id,
            "document_name": document_name,
            "chunk_text": chunk_text,
            "similarity": round(similarity_score, 4),
        }
        chunks.append(chunk_record)
        context_parts.append(f"[Source: {document_name}]\n{chunk_text}")
        sources.append(
            {
                "id": chunk_id,
                "document_name": document_name,
                "similarity": round(similarity_score, 4),
            }
        )

    chunks = sorted(chunks, key=lambda item: item["similarity"], reverse=True)
    context_parts = [
        f"[Source: {chunk['document_name']}]\n{chunk['chunk_text']}" for chunk in chunks
    ]
    sources = [
        {
            "id": chunk["id"],
            "document_name": chunk["document_name"],
            "similarity": chunk["similarity"],
            "text": chunk["chunk_text"],
        }
        for chunk in chunks
    ]

    return {"context": "\n\n".join(context_parts), "sources": sources, "chunks": chunks}


def _extract_answer_text(response: Any) -> str:
    answer_text = getattr(response, "content", str(response))
    if isinstance(answer_text, list):
        answer_text = "".join(
            part.get("text", "") for part in answer_text if isinstance(part, dict)
        )
    return str(answer_text).strip()


def _is_missing_answer(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return (
        "cannot determine" in lowered
        or "not enough information" in lowered
        or "not provided" in lowered
        or "not present in the context" in lowered
        or "cannot answer based on the provided data" in lowered
    )


def answer_question(question: str, top_k: int = 5) -> dict[str, Any]:
    """Ground an answer using all retrieved pgvector chunks as context, ordered by similarity."""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    retrieval = retrieve_relevant_chunks(question, top_k=top_k)
    chunks = retrieval.get("chunks") or []
    if not chunks:
        return {
            "answer": "I could not find relevant information in the indexed database for this question.",
            "context": "",
            "sources": [],
        }

    combined_context = "\n\n".join(
        f"[Source: {chunk['document_name']}]\n{chunk['chunk_text']}" for chunk in chunks
    )
    llm = build_bedrock_llm()

    prompt = f"""
You are a careful document question-answering assistant.
Answer using only the context below. Your answer MUST BE short and precise.
If the answer is not present in the context, say that it cannot be determined from the provided data.
The strongest match is listed first in the context below.

Context:
{combined_context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    answer_text = _extract_answer_text(response)

    return {
        "answer": answer_text,
        "context": combined_context,
        "sources": retrieval.get("sources", []),
        "source_texts": [
            source.get("text", "") for source in retrieval.get("sources", [])
        ],
    }
