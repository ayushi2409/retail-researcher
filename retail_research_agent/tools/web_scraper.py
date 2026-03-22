"""HTTP fetch + BeautifulSoup text extraction with optional async batching."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import aiohttp
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from config.settings import Settings, get_settings
from utils.helpers import normalize_whitespace
from utils.logger import get_logger, log_step

logger = get_logger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RetailResearchBot/1.0; +https://example.com/bot)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


def _visible_text_from_soup(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    for sel in ("nav", "footer", "header", "aside", "form"):
        for el in soup.find_all(sel):
            el.decompose()
    text = soup.get_text(separator="\n")
    return normalize_whitespace(text)


def _html_to_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article") or soup.find("main") or soup.body or soup
    text = _visible_text_from_soup(BeautifulSoup(str(article), "lxml"))
    if len(text) < 80:
        text = _visible_text_from_soup(soup)
    return text[:25_000]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=15),
    reraise=False,
)
def _fetch(url: str, timeout: int) -> Optional[str]:
    r = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def scrape_url(url: str, settings: Optional[Settings] = None) -> str:
    """
    Download HTML and return cleaned main text.

    On failure returns a short diagnostic string (never raises to callers).
    """
    settings = settings or get_settings()
    if not url or urlparse(url).scheme not in ("http", "https"):
        return "[scrape skipped: invalid url]"
    log_step(logger, "scrape_start", details={"url": url[:200]})
    try:
        html = _fetch(url, settings.scrape_timeout_sec)
        if not html:
            return "[scrape failed: empty response]"
        clipped = _html_to_clean_text(html)
        log_step(logger, "scrape_ok", details={"chars": len(clipped)})
        return clipped
    except Exception as exc:
        log_step(logger, "scrape_error", status="error", details={"error": str(exc)[:200]})
        return f"[scrape failed for {url}: {exc}]"


async def _fetch_async(
    session: aiohttp.ClientSession,
    url: str,
    timeout_sec: int,
) -> Tuple[str, Optional[str]]:
    try:
        to = aiohttp.ClientTimeout(total=timeout_sec)
        async with session.get(url, timeout=to) as resp:
            resp.raise_for_status()
            text = await resp.text()
            return url, text
    except Exception as exc:
        log_step(logger, "scrape_error", status="error", details={"url": url[:120], "error": str(exc)[:200]})
        return url, None


async def _scrape_urls_async(urls: List[str], settings: Settings) -> Dict[str, str]:
    seen: Set[str] = set()
    to_fetch: List[str] = []
    for u in urls:
        if not u or u in seen:
            continue
        if urlparse(u).scheme not in ("http", "https"):
            continue
        seen.add(u)
        to_fetch.append(u)
        if len(to_fetch) >= settings.scrape_max_urls:
            break

    log_step(logger, "scrape_async_batch", details={"count": len(to_fetch)})
    out: Dict[str, str] = {}
    sem = asyncio.Semaphore(8)

    async with aiohttp.ClientSession(headers=_DEFAULT_HEADERS) as session:

        async def bounded(url: str) -> Tuple[str, Optional[str]]:
            async with sem:
                return await _fetch_async(session, url, settings.scrape_timeout_sec)

        results = await asyncio.gather(*(bounded(u) for u in to_fetch))

    for url, html in results:
        if html:
            out[url] = _html_to_clean_text(html)
            log_step(logger, "scrape_ok", details={"url": url[:120], "chars": len(out[url])})
        else:
            out[url] = f"[scrape failed for {url}]"
    return out


def scrape_urls(urls: List[str], settings: Optional[Settings] = None) -> Dict[str, str]:
    """Scrape unique URLs up to settings.scrape_max_urls (sync or async)."""
    settings = settings or get_settings()
    if settings.enable_async_scrape:
        return asyncio.run(_scrape_urls_async(urls, settings))

    seen: Set[str] = set()
    out: Dict[str, str] = {}
    for u in urls:
        if not u or u in seen:
            continue
        seen.add(u)
        if len(out) >= settings.scrape_max_urls:
            break
        out[u] = scrape_url(u, settings)
    return out
