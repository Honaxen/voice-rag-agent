"""
tests/test_pipeline.py — Unit tests for pipeline components.

Tests the deterministic parts without needing Whisper, Ollama, or microphone.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.ingestion import clean_text, chunk_document, ingest
from agent.retrieval import VectorStore, _rrf


# ─────────────────────────────────────────────
# Ingestion tests
# ─────────────────────────────────────────────

class TestIngestion:
    def test_clean_text_removes_null_bytes(self):
        result = clean_text("hello\x00world")
        assert "\x00" not in result

    def test_clean_text_normalizes_whitespace(self):
        result = clean_text("hello   world")
        assert "  " not in result

    def test_chunk_document_basic(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunk_document(text, sentences_per_chunk=2, min_length=20)
        assert len(chunks) > 0

    def test_chunk_document_min_length(self):
        text = "A. B. C. D."
        chunks = chunk_document(text, sentences_per_chunk=2, min_length=50)
        # Short chunks should be filtered
        for chunk in chunks:
            assert len(chunk) >= 50

    def test_ingest_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("This is a test document. It has several sentences. We will chunk it. This is the fourth sentence.")
            path = f.name
        try:
            chunks = ingest(path)
            assert isinstance(chunks, list)
        finally:
            os.unlink(path)

    def test_ingest_unsupported_format(self):
        import pytest
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            try:
                ingest(path)
                assert False, "Should have raised ValueError"
            except ValueError:
                pass
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────
# Retrieval tests
# ─────────────────────────────────────────────

class TestRRF:
    def test_rrf_combines_results(self):
        bm25 = [(0, 1.0), (1, 0.8), (2, 0.5)]
        faiss = [(1, 0.1), (0, 0.2), (3, 0.3)]
        result = _rrf(bm25, faiss)
        indices = [idx for idx, _ in result]
        # Both 0 and 1 appeared in both lists — should rank highest
        assert 0 in indices
        assert 1 in indices

    def test_rrf_scores_are_positive(self):
        bm25 = [(0, 1.0), (1, 0.5)]
        faiss = [(0, 0.2), (2, 0.1)]
        result = _rrf(bm25, faiss)
        for _, score in result:
            assert score > 0

    def test_rrf_deduplicates(self):
        bm25 = [(0, 1.0), (0, 0.5)]
        faiss = [(0, 0.2)]
        result = _rrf(bm25, faiss)
        indices = [idx for idx, _ in result]
        assert indices.count(0) == 1


class TestVectorStore:
    def test_build_and_retrieve(self):
        vs = VectorStore()
        chunks = [
            "Paris is the capital of France.",
            "The Eiffel Tower is in Paris.",
            "Berlin is the capital of Germany.",
        ]
        vs.build(chunks)
        results = vs.retrieve("What is the capital of France?", top_k=2)
        assert len(results) == 2
        assert any("Paris" in r for r in results)

    def test_size_property(self):
        vs = VectorStore()
        vs.build(["chunk one", "chunk two", "chunk three"])
        assert vs.size == 3

    def test_retrieve_before_build_raises(self):
        vs = VectorStore()
        try:
            vs.retrieve("query")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass