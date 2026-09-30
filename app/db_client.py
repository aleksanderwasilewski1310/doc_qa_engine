import os
import psycopg
from datetime import datetime
from typing import List, Optional


def _get_db_config():
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5433)),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "dbname": os.getenv("POSTGRES_DB", "postgres"),
    }


def get_connection():
    cfg = _get_db_config()
    return psycopg.connect(**cfg)


def insert_chunk(
    document_name: str,
    chunk_text: str,
    distance: Optional[float] = None,
    timestamp: Optional[datetime] = None,
    embedding: Optional[List[float]] = None,
) -> int:
    """Insert a single chunk into the `doc_chunks` table.

    Returns the inserted row id.
    """
    ts = timestamp if timestamp is not None else datetime.utcnow()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if embedding is not None:
                # Convert embedding list to Postgres vector literal string and cast to vector
                emb_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                cur.execute(
                    "INSERT INTO doc_chunks (document_name, timestamp, chunk_text, distance, embedding) VALUES (%s, %s, %s, %s, %s::vector) RETURNING id",
                    (document_name, ts, chunk_text, distance, emb_str),
                )
            else:
                cur.execute(
                    "INSERT INTO doc_chunks (document_name, timestamp, chunk_text, distance) VALUES (%s, %s, %s, %s) RETURNING id",
                    (document_name, ts, chunk_text, distance),
                )
            inserted = cur.fetchone()[0]
        conn.commit()
        return inserted
    finally:
        conn.close()


def insert_chunks(chunks: List[dict]) -> None:
    """Insert multiple chunks. Each chunk is a dict with keys:
    - document_name (str)
    - chunk_text (str)
    - distance (optional float)
    - timestamp (optional datetime)
    """
    if not chunks:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for c in chunks:
                ts = c.get("timestamp") or datetime.utcnow()
                embedding = c.get("embedding")
                if embedding is not None:
                    emb_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                    cur.execute(
                        "INSERT INTO doc_chunks (document_name, timestamp, chunk_text, distance, embedding) VALUES (%s,%s,%s,%s,%s::vector)",
                        (
                            c["document_name"],
                            ts,
                            c["chunk_text"],
                            c.get("distance"),
                            emb_str,
                        ),
                    )
                else:
                    cur.execute(
                        "INSERT INTO doc_chunks (document_name, timestamp, chunk_text, distance) VALUES (%s,%s,%s,%s)",
                        (c["document_name"], ts, c["chunk_text"], c.get("distance")),
                    )
        conn.commit()
    finally:
        conn.close()


__all__ = ["get_connection", "insert_chunk", "insert_chunks"]
