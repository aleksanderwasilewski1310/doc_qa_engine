"""Database connection and document-chunk persistence helpers."""

import os
from datetime import UTC, datetime

import psycopg


def _get_db_config():
    """Build PostgreSQL connection settings from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5433")),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "dbname": os.getenv("DB_NAME", "postgres"),
    }


def get_connection():
    """Open and return a PostgreSQL connection using the configured settings."""
    cfg = _get_db_config()
    return psycopg.connect(**cfg)


def insert_chunk(
    document_name: str,
    chunk_text: str,
    distance: float | None = None,
    timestamp: datetime | None = None,
    embedding: list[float] | None = None,
) -> int:
    """Insert one document chunk into the ``doc_chunks`` table.

    Args:
        document_name: Name of the source document.
        chunk_text: Text content of the chunk.
        distance: Optional retrieval distance associated with the chunk.
        timestamp: Optional timestamp to store; defaults to the current UTC time.
        embedding: Optional vector embedding to store alongside the chunk.

    Returns:
        The ID of the inserted row.
    """
    ts = timestamp if timestamp is not None else datetime.now(UTC)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if embedding is not None:
                # Convert embedding list to Postgres vector literal string and cast to vector
                emb_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                cur.execute(
                    "INSERT INTO doc_chunks (document_name, timestamp, chunk_text," \
                    " distance, embedding) VALUES (%s, %s, %s, %s, %s::vector) RETURNING id",
                    (document_name, ts, chunk_text, distance, emb_str),
                )
            else:
                cur.execute(
                    "INSERT INTO doc_chunks (document_name, timestamp, chunk_text, distance) " \
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (document_name, ts, chunk_text, distance),
                )
            inserted = cur.fetchone()[0]
        conn.commit()
        return inserted
    finally:
        conn.close()


def insert_chunks(chunks: list[dict]) -> None:
    """Insert multiple chunks into the ``doc_chunks`` table.

    Each chunk dictionary must contain ``document_name`` and ``chunk_text``.
    It may also contain ``distance``, ``timestamp``, and ``embedding``.

    Args:
        chunks: Chunk records to insert. An empty list is ignored.
    """
    if not chunks:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for c in chunks:
                ts = c.get("timestamp") or datetime.now(UTC)
                embedding = c.get("embedding")
                if embedding is not None:
                    emb_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                    cur.execute(
                        "INSERT INTO doc_chunks (document_name, timestamp, chunk_text," \
                        " distance, embedding) VALUES (%s,%s,%s,%s,%s::vector)",
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
                        "INSERT INTO doc_chunks (document_name, timestamp, chunk_text," \
                        " distance) VALUES (%s,%s,%s,%s)",
                        (c["document_name"], ts, c["chunk_text"], c.get("distance")),
                    )
        conn.commit()
    finally:
        conn.close()


__all__ = ["get_connection", "insert_chunk", "insert_chunks"]
