"""Tests for the PDF chunking workflow.

These tests cover the public contract of chunk_pdf: validating file inputs,
ensuring PDF pages are split into chunks, and verifying that chunk metadata is
attached to each returned LangChain document.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from app import chunking


class TestChunkPdf(unittest.TestCase):
    def test_chunk_pdf_splits_loaded_documents_and_adds_metadata(self):
        """Chunked documents should keep page content and receive chunk metadata."""

        class FakeLoader:
            def __init__(self, path):
                self.path = path

            def load(self):
                return [
                    Document(
                        page_content="First paragraph.\n\nSecond paragraph.",
                        metadata={"source": self.path, "page": 1},
                    ),
                    Document(
                        page_content="Third paragraph.",
                        metadata={"source": self.path, "page": 2},
                    ),
                ]

        class FakeTextSplitter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def split_documents(self, docs):
                return [
                    Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                    for doc in docs
                ]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(b"%PDF-1.4\n")
            temp_path = Path(temp_pdf.name)

        try:
            with (
                patch.object(chunking, "PyPDFLoader", FakeLoader),
                patch.object(
                    chunking, "RecursiveCharacterTextSplitter", FakeTextSplitter
                ),
            ):
                chunks = chunking.chunk_pdf(temp_path)

            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0].metadata["chunk_id"], 0)
            self.assertEqual(chunks[0].metadata["total_chunks"], 2)
            self.assertEqual(chunks[0].metadata["file_name"], temp_path.name)
            self.assertIn("First paragraph", chunks[0].page_content)
            self.assertIn("Third paragraph", chunks[1].page_content)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_chunk_pdf_rejects_non_pdf_files(self):
        """Non-PDF uploads should be rejected before loading starts."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_txt:
            temp_path = Path(temp_txt.name)

        try:
            with self.assertRaises(ValueError):
                chunking.chunk_pdf(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_chunk_pdf_rejects_missing_files(self):
        """A missing PDF path should raise FileNotFoundError immediately."""
        missing_path = Path("/definitely/missing/file.pdf")

        with self.assertRaises(FileNotFoundError):
            chunking.chunk_pdf(missing_path)

    def test_main_propagates_database_write_failure(self):
        chunk = Document(
            page_content="A chunk of text.",
            metadata={"file_name": "sample.pdf"},
        )

        with (
            patch.object(chunking, "chunk_pdf", return_value=[chunk]),
            patch.object(chunking, "embed_texts", None),
            patch.object(
                chunking, "insert_chunks", side_effect=RuntimeError("database down")
            ),
            self.assertRaisesRegex(RuntimeError, "database down"),
        ):
            chunking.main("sample.pdf", 10.0)


if __name__ == "__main__":
    unittest.main()
