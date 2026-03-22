"""Embedding-based deduplication for scraped page blobs."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from config.settings import Settings, get_settings
from memory.embeddings_factory import build_embeddings
from utils.helpers import chunk_for_embedding, deduplicate_by_embedding_similarity
from utils.logger import get_logger, log_step

logger = get_logger(__name__)

_URL_HEAD = re.compile(r"###\s*URL:\s*(https?://\S+)", re.MULTILINE)


def _split_scraped_corpus(raw: str) -> List[Tuple[str, str]]:
    """Return list of (url, body) using ### URL: markers; fallback to paragraph chunks."""
    matches = list(_URL_HEAD.finditer(raw))
    if matches:
        out: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            url = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            body = raw[start:end].strip()
            if body:
                out.append((url, body))
        return out or [("", raw)]

    paras = [p.strip() for p in re.split(r"\n{2,}", raw) if len(p.strip()) > 80]
    return [("", p) for p in paras[:40]]


def deduplicate_scraped_corpus(raw: str, settings: Optional[Settings] = None) -> str:
    """
    Remove near-duplicate sections using embedding similarity.

    Preserves URL headers when present.
    """
    settings = settings or get_settings()
    blocks = _split_scraped_corpus(raw)
    if len(blocks) <= 1:
        return raw

    log_step(logger, "dedup_start", details={"blocks": len(blocks)})
    embedder = build_embeddings(settings)

    labeled: List[Tuple[str, str, str]] = []
    for url, body in blocks:
        text = chunk_for_embedding(body, max_chars=6000)
        header = f"### URL: {url}\n" if url else ""
        labeled.append((header, text, url))

    vectors = embedder.embed_documents([t for _, t, _ in labeled])
    items = [(f"{h}{t}", v) for (h, t, _), v in zip(labeled, vectors)]
    kept = deduplicate_by_embedding_similarity(items, settings.dedup_similarity_threshold)

    merged_parts: List[str] = []
    for content, _ in kept:
        merged_parts.append(content.strip())
    result = "\n\n---\n\n".join(merged_parts)
    log_step(logger, "dedup_done", details={"kept": len(kept), "dropped": len(items) - len(kept)})
    return result
