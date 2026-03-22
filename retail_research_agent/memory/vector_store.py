"""Vector backends for report chunks: Chroma (default) or FAISS (toggle via settings)."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document

from config.settings import Settings, get_settings
from memory.embeddings_factory import build_embeddings
from utils.helpers import chunk_for_embedding
from utils.logger import get_logger

logger = get_logger(__name__)


class _VectorBackend(ABC):
    @abstractmethod
    def add_report(
        self,
        report_text: str,
        *,
        user_query: str,
        title: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        pass


class _ChromaBackend(_VectorBackend):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = build_embeddings(settings)
        self._store = Chroma(
            collection_name="retail_reports",
            embedding_function=self._embeddings,
            persist_directory=str(settings.chroma_persist_dir),
        )

    def add_report(
        self,
        report_text: str,
        *,
        user_query: str,
        title: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        report_id = str(uuid.uuid4())
        text = chunk_for_embedding(report_text, max_chars=12000)
        doc = Document(
            page_content=text,
            metadata={
                "report_id": report_id,
                "user_query": user_query[:2000],
                "title": title[:500],
                **(extra_metadata or {}),
            },
        )
        self._store.add_documents([doc])
        try:
            if hasattr(self._store, "persist"):
                self._store.persist()
        except Exception as exc:
            logger.warning("chroma_persist_skipped: %s", str(exc))
        logger.info(
            "vector_store_add",
            extra={"structured": {"backend": "chroma", "report_id": report_id, "chars": len(text)}},
        )
        return report_id

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self._store.similarity_search(query, k=k)


class _FaissBackend(_VectorBackend):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._path = settings.faiss_index_dir
        self._path.mkdir(parents=True, exist_ok=True)
        self._embeddings = build_embeddings(settings)
        index_file = self._path / "index.faiss"
        if index_file.exists():
            self._store = FAISS.load_local(
                str(self._path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self._store = None

    def add_report(
        self,
        report_text: str,
        *,
        user_query: str,
        title: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        report_id = str(uuid.uuid4())
        text = chunk_for_embedding(report_text, max_chars=12000)
        doc = Document(
            page_content=text,
            metadata={
                "report_id": report_id,
                "user_query": user_query[:2000],
                "title": title[:500],
                **(extra_metadata or {}),
            },
        )
        if self._store is None:
            self._store = FAISS.from_documents([doc], self._embeddings)
        else:
            self._store.add_documents([doc])
        self._store.save_local(str(self._path))
        logger.info(
            "vector_store_add",
            extra={"structured": {"backend": "faiss", "report_id": report_id, "chars": len(text)}},
        )
        return report_id

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)


class ReportVectorStore:
    """Facade: Chroma or FAISS based on ``Settings.vector_backend``."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        if self._settings.vector_backend == "faiss":
            self._backend: _VectorBackend = _FaissBackend(self._settings)
        else:
            self._backend = _ChromaBackend(self._settings)

    def add_report(
        self,
        report_text: str,
        *,
        user_query: str,
        title: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._backend.add_report(
            report_text,
            user_query=user_query,
            title=title,
            extra_metadata=extra_metadata,
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self._backend.similarity_search(query, k=k)
