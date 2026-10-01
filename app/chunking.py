"""PDF Chunking Module for RAG Ingestion Pipelines.

This module provides functionality to load a PDF file, process its content,
and split it into semantically coherent chunks using LangChain's recursive
character text splitter.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Import DB client using a relative import when running as a package,
# falling back to absolute import when executed as a script.
try:
    from .db_client import insert_chunks
except ImportError as err:
    logger.error("Error importing db_client: %s", err)
    try:
        from db_client import insert_chunks
    except ImportError as err:
        logger.error("Error importing insert_chunks: %s", err)

        # Fallback dummy function if db_client is not present during standalone execution/testing
        def insert_chunks(records: list[dict[str, Any]]) -> None:
            pass


# Suppress noisy DeprecationWarnings from langchain_community which may be
# promoted to errors in some environments.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"langchain_community.*"
)

# Prefer langchain_community or modern LangChain integration modules
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError as err:
    logger.error("Error importing PyPDFLoader from langchain_community: %s", err)
    try:
        from langchain_community import PyPDFLoader
    except ImportError as err:
        logger.error("Error importing PyPDFLoader from langchain_community: %s", err)
        raise ImportError(
            "PyPDFLoader could not be imported. Please install langchain-community and pypdf."
        ) from err

# Try to import embedding helper; if missing, continue without embeddings
try:
    from .embeddings import embed_texts
except ImportError as err:
    logger.error("Error importing embed_texts: %s", err)
    try:
        from embeddings import embed_texts
    except ImportError as err:
        logger.error("Error importing embed_texts: %s", err)
        embed_texts = None


def chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 300,
) -> list[Document]:
    """Loads a PDF document and splits its content into manageable chunks.

    Args:
        pdf_path (str | Path): The filesystem path to the PDF file.
        chunk_size (int, optional): The maximum size of each chunk in characters.
            Defaults to 800.
        chunk_overlap (int, optional): The character overlap between consecutive chunks
            to preserve context across boundaries. Defaults to 150.

    Returns:
        List[Document]: A list of LangChain Document objects containing chunked
            text and enriched metadata (source path, page numbers, etc.).

    Raises:
        FileNotFoundError: If the provided PDF file path does not exist.
        ValueError: If the input file is not a valid PDF.
    """
    path = Path(pdf_path)

    # Validate file existence
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at path: {path.resolve()}")

    # Validate file extension
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, but received: {path.suffix}")

    # Step 1: Load PDF pages using PyPDFLoader
    loader = PyPDFLoader(str(path))
    raw_documents = loader.load()

    # Step 2: Configure text splitter tailored for document structure
    # Hierarchy of separators maintains paragraph and sentence integrity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        # Increasing overlap prevents context loss at chunk boundaries
        chunk_overlap=max(chunk_overlap, 250),
        length_function=len,
        is_separator_regex=True,  # Enable regex!
        separators=[
            r"\n(?=[I|V|X]+\.\s)",  # Splits before Roman numeral headers, e.g., \nII.
            r"\n(?=§\s*\d+)",  # Splits before section symbols, e.g., \n§ 1
            "\n\n",
            "\n",
            r"\. ",
            " ",
            "",
        ],
    )

    # Step 3: Split documents into chunks while preserving and expanding metadata
    chunked_documents = text_splitter.split_documents(raw_documents)

    # Step 4: Enrich metadata for Vector Store / Hybrid Search usage
    for idx, doc in enumerate(chunked_documents):
        doc.metadata.update(
            {
                "chunk_id": idx,
                "total_chunks": len(chunked_documents),
                "file_name": path.name,
            }
        )

    return chunked_documents


def main(pdf_path: str, race_distance: float) -> None:
    """Example execution entry point."""
    sample_pdf_path = pdf_path

    try:
        # Process the PDF document
        chunks = chunk_pdf(
            pdf_path=sample_pdf_path,
            chunk_size=600,
            chunk_overlap=100,
        )
        for chunk in chunks:
            chunk.metadata["race_distance"] = race_distance

        logger.info("Successfully processed PDF. Created %d chunks.", len(chunks))

    except FileNotFoundError as err:
        logger.error("File not found: %s", err)
    except ValueError as err:
        logger.error("Validation error: %s", err)
    except Exception:
        logger.exception("An unexpected error occurred during processing")

    else:
        # Executed ONLY when PDF processing succeeded without exceptions
        try:
            records = [
                {
                    "document_name": c.metadata.get("file_name", Path(pdf_path).name),
                    "chunk_text": c.page_content,
                    "distance": c.metadata.get("race_distance"),
                }
                for c in chunks
            ]

            # If embedding support is available, compute embeddings in batch and attach to records
            if embed_texts is not None and records:
                try:
                    texts = [r["chunk_text"] for r in records]
                    vectors = embed_texts(texts)
                    # Ensure vectors length matches
                    if vectors and len(vectors) == len(records):
                        for r, v in zip(records, vectors):
                            r["embedding"] = v
                    else:
                        logger.warning(
                            "Embedding count mismatch: %d texts -> %d vectors",
                            len(records),
                            len(vectors) if vectors else 0,
                        )
                except Exception:
                    logger.exception(
                        "Failed to compute embeddings; continuing without them"
                    )

            if records:
                insert_chunks(records)
                logger.info("Inserted %d chunks into the database.", len(records))
        except Exception:
            logger.exception("Failed to write chunks to DB")


if __name__ == "__main__":
    # Configure logging output for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    main(
        pdf_path=r"C:\\Users\\awasi\\Downloads\\Regulamin-Nice-To-Fit-You-Warszawskiej-Dychy_2026.pdf",
        race_distance=10.0,
    )
