"""Build LangChain embedding backends (OpenAI, local Hugging Face, or Ollama)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

if TYPE_CHECKING:
    from config.settings import Settings


def build_embeddings(settings: "Settings") -> Embeddings:
    """
    Create embeddings for Chroma/FAISS and corpus deduplication.

    Use ``embedding_provider=auto`` (default) to pick OpenAI when ``OPENAI_API_KEY`` is set,
    otherwise local Hugging Face (no API key; first run downloads the model).
    """
    provider = settings.resolved_embedding_provider()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER is openai or auto with key set.")
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.huggingface_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unknown embedding provider: {provider}")
