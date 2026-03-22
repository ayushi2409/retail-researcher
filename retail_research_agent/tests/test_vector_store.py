"""Vector backend selection (no live OpenAI calls for empty FAISS index)."""

from __future__ import annotations

from pathlib import Path

from config.settings import get_settings
from memory.vector_store import ReportVectorStore


def test_faiss_backend_empty_index_no_crash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "faiss_idx"))
    get_settings.cache_clear()

    store = ReportVectorStore()
    assert store.similarity_search("retail", k=3) == []
