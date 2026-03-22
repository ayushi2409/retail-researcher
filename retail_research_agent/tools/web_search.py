"""Tavily / Serper web search with retries, caching, and empty-result handling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from config.settings import Settings, get_settings
from utils.helpers import hash_text, score_source_credibility
from utils.logger import get_logger, log_step

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Normalized search hit."""

    title: str
    url: str
    snippet: str
    credibility: float
    credibility_note: str


class WebSearchService:
    """Search provider abstraction (Tavily preferred, Serper fallback)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._settings.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Any:
        return self._settings.cache_dir / f"search_{key}.json"

    def _read_cache(self, key: str) -> Optional[List[dict]]:
        if not self._settings.enable_cache:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, key: str, payload: List[dict]) -> None:
        if not self._settings.enable_cache:
            return
        path = self._cache_path(key)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query: str) -> List[SearchResult]:
        """
        Run web search with retries; return top results with credibility scores.

        Handles empty API keys and low-quality/empty payloads gracefully.
        """
        log_step(logger, "web_search_start", details={"query_preview": query[:120]})
        cache_key = hash_text(f"{query}|tavily|serper")
        cached = self._read_cache(cache_key)
        if cached is not None:
            log_step(logger, "web_search_cache_hit", details={"key": cache_key[:16]})
            return [SearchResult(**row) for row in cached]

        raw: List[dict] = []
        if self._settings.tavily_api_key:
            raw = self._search_tavily(query)
        elif self._settings.serper_api_key:
            raw = self._search_serper(query)
        else:
            log_step(logger, "web_search_no_provider", status="degraded")
            return [
                SearchResult(
                    title="Configuration required",
                    url="https://example.invalid/configure-search",
                    snippet=(
                        "No TAVILY_API_KEY or SERPER_API_KEY configured. "
                        "Add one to .env to enable live web search."
                    ),
                    credibility=0.0,
                    credibility_note="missing_api_key",
                )
            ]

        if not raw:
            log_step(logger, "web_search_empty", status="warn")
            return [
                SearchResult(
                    title="No results",
                    url="",
                    snippet="Search returned no usable results; widen the query or retry.",
                    credibility=0.0,
                    credibility_note="empty",
                )
            ]

        results: List[SearchResult] = []
        for item in raw[: self._settings.search_max_results]:
            url = item.get("url") or ""
            title = item.get("title") or "Untitled"
            snippet = item.get("snippet") or item.get("content") or ""
            cred, note = score_source_credibility(url)
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet[: self._settings.search_snippet_max_chars],
                    credibility=cred,
                    credibility_note=note,
                )
            )

        serializable = [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "credibility": r.credibility,
                "credibility_note": r.credibility_note,
            }
            for r in results
        ]
        self._write_cache(cache_key, serializable)
        log_step(logger, "web_search_done", details={"count": len(results)})
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=20),
        reraise=True,
    )
    def _search_tavily(self, query: str) -> List[dict]:
        from tavily import TavilyClient

        client = TavilyClient(api_key=self._settings.tavily_api_key)
        resp = client.search(
            query=query,
            max_results=self._settings.search_max_results,
            search_depth="advanced",
        )
        out: List[dict] = []
        for r in resp.get("results") or []:
            out.append(
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": r.get("content") or r.get("snippet"),
                }
            )
        return out

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=20),
        reraise=True,
    )
    def _search_serper(self, query: str) -> List[dict]:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self._settings.serper_api_key or "", "Content-Type": "application/json"}
        body = {"q": query, "num": self._settings.search_max_results}
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic") or []
        out: List[dict] = []
        for item in organic:
            out.append(
                {
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                }
            )
        return out


def format_search_results_for_prompt(
    results: List[SearchResult],
    *,
    max_total_chars: int | None = None,
) -> str:
    """Render search results as markdown for agent context."""
    lines: List[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"{i}. **{r.title}** (credibility {r.credibility:.2f} — {r.credibility_note})\n"
            f"   - URL: {r.url}\n"
            f"   - Snippet: {r.snippet}\n"
        )
    out = "\n".join(lines)
    if max_total_chars is not None and len(out) > max_total_chars:
        note = "\n\n[TRUNCATED: search output capped; URLs/snippets above are still valid.]\n"
        keep = max_total_chars - len(note)
        if keep < 200:
            keep = max_total_chars
            note = ""
        out = out[:keep] + note
    return out
