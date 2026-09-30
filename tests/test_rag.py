import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import rag


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content="I cannot determine the answer from the provided context."
            )
        return SimpleNamespace(content="The answer is in the second chunk.")


class TestRagFallback(unittest.TestCase):
    @patch("app.rag.build_bedrock_llm")
    @patch("app.rag.retrieve_relevant_chunks")
    def test_answer_question_goes_to_next_chunk_when_first_is_not_relevant(
        self, mock_retrieve, mock_llm_factory
    ):
        mock_llm = FakeLLM()
        mock_llm_factory.return_value = mock_llm
        mock_retrieve.return_value = {
            "context": "first chunk\n\nsecond chunk",
            "sources": [
                {"document_name": "doc1.pdf", "similarity": 0.90},
                {"document_name": "doc1.pdf", "similarity": 0.85},
            ],
            "chunks": [
                {
                    "document_name": "doc1.pdf",
                    "chunk_text": "first chunk",
                    "similarity": 0.90,
                },
                {
                    "document_name": "doc1.pdf",
                    "chunk_text": "second chunk",
                    "similarity": 0.85,
                },
            ],
        }

        result = rag.answer_question("What is the answer?", top_k=5)

        self.assertEqual(result["answer"], "The answer is in the second chunk.")
        self.assertEqual(len(mock_llm.calls), 2)


if __name__ == "__main__":
    unittest.main()
