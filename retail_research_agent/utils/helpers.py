"""General helpers: hashing, similarity, deduplication, source scoring."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Domains often associated with higher editorial standards (heuristic, not exhaustive).
_TRUSTED_TLDS = frozenset({".gov", ".edu", ".int"})
_TRUSTED_KEYWORDS = (
    "reuters",
    "bloomberg",
    "ft.com",
    "economist",
    "mckinsey",
    "kpmg",
    "deloitte",
    "pwc",
    "ey.com",
    "nielsen",
    "statista",
    "ibef",
    "retailwire",
    "forbes",
    "wsj",
)


def hash_text(text: str) -> str:
    """Stable SHA256 hex digest for cache keys."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for two equal-length vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def score_source_credibility(url: str) -> Tuple[float, str]:
    """
    Return (0..1 score, short rationale) using lightweight heuristics.

    This is not a substitute for human review; it prioritizes known-quality patterns.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return 0.35, "unparseable_url"

    host = (parsed.hostname or "").lower()
    if not host:
        return 0.35, "no_host"

    score = 0.55
    reasons: List[str] = ["baseline"]

    if any(host.endswith(t) for t in _TRUSTED_TLDS):
        score += 0.2
        reasons.append("trusted_tld")

    if any(k in host for k in _TRUSTED_KEYWORDS):
        score += 0.15
        reasons.append("recognized_brand")

    if "wikipedia.org" in host:
        score += 0.05
        reasons.append("encyclopedic")

    # Penalize obvious UGC / aggregators lightly (still may be useful).
    if any(x in host for x in ("reddit", "facebook", "tiktok", "pinterest")):
        score -= 0.15
        reasons.append("ugc_platform")

    score = max(0.0, min(1.0, score))
    return score, ",".join(reasons)


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace for comparison and display."""
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def deduplicate_by_embedding_similarity(
    items: Iterable[Tuple[str, Sequence[float]]],
    threshold: float,
) -> List[Tuple[str, Sequence[float]]]:
    """
    Greedy clustering by embedding similarity: keep first item, drop near-duplicates.

    items: iterable of (payload_text, embedding_vector)
    """
    kept: List[Tuple[str, Sequence[float]]] = []
    for text, emb in items:
        if not text or not emb:
            continue
        is_dup = False
        for _, kemb in kept:
            if cosine_similarity(emb, kemb) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append((text, emb))
    return kept


def chunk_for_embedding(text: str, max_chars: int = 6000) -> str:
    """Truncate very long blobs before embedding API calls."""
    t = normalize_whitespace(text)
    if len(t) <= max_chars:
        return t
    return t[:max_chars]
