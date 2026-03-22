"""Structured retail report schema and light markdown validation."""

from __future__ import annotations

import re
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator


class RetailReport(BaseModel):
    """Expected sections for the final client-facing research report."""

    title: str = Field(..., min_length=1, max_length=500)
    summary: str = Field(..., min_length=10)
    key_insights: List[str] = Field(..., min_length=1)
    market_trends: str = Field(..., min_length=5)
    competitor_analysis: str = Field(..., min_length=5)
    risks: str = Field(..., min_length=5)
    opportunities: str = Field(..., min_length=5)
    sources: List[str] = Field(..., min_length=1)

    @field_validator("sources", mode="before")
    @classmethod
    def strip_sources(cls, v: object) -> object:
        if isinstance(v, list):
            return [s.strip() for s in v if isinstance(s, str) and s.strip()]
        return v

    @field_validator("key_insights", mode="before")
    @classmethod
    def strip_insights(cls, v: object) -> object:
        if isinstance(v, list):
            return [s.strip() for s in v if isinstance(s, str) and s.strip()]
        return v


_HEADER_RE = re.compile(
    r"^#+\s*(.+?)\s*$",
    re.MULTILINE,
)


def _split_markdown_sections(markdown: str) -> Dict[str, str]:
    """
    Split markdown on ATX headers (# …) into normalized section keys.

    Keys are lowercased alphanumeric; first H1 is treated as title line only.
    """
    text = markdown.strip()
    if not text:
        return {}

    parts = _HEADER_RE.split(text)
    if len(parts) < 3:
        return {"body": text}

    # parts[0] may be preamble; parts[1] first title, parts[2] first body, ...
    sections: Dict[str, str] = {}
    preamble = parts[0].strip()
    i = 1
    while i + 1 < len(parts):
        raw_title = parts[i].strip()
        body = parts[i + 1].strip()
        key = re.sub(r"[^a-z0-9]+", "_", raw_title.lower()).strip("_") or "section"
        if key == "title" or (not sections and i == 1):
            sections["title"] = raw_title
            if body:
                sections["title_body"] = body
        else:
            sections[key] = body
        i += 2

    if preamble and "preamble" not in sections:
        sections["_preamble"] = preamble
    return sections


def _bullets_to_list(block: str) -> List[str]:
    lines = []
    for line in block.splitlines():
        line = re.sub(r"^[\s>*\-•\d.)]+", "", line).strip()
        if len(line) > 3:
            lines.append(line)
    return lines or [block.strip()[:2000]]


def _urls_from_block(block: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"https?://[^\s)\]>]+", block)))


def parse_and_validate_retail_report(markdown: str) -> RetailReport:
    """
    Best-effort parse of writer markdown into ``RetailReport``.

    Raises ``ValidationError`` if required content is still missing after heuristics.
    """
    sec = _split_markdown_sections(markdown)

    title = sec.get("title", "").strip()
    if not title:
        m = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        title = m.group(1).strip() if m else "Retail research report"

    def pick(*keys: str, default: str = "") -> str:
        for k in keys:
            v = sec.get(k)
            if v and len(v.strip()) >= 5:
                return v.strip()
        return default

    summary = pick("summary", "executive_summary", "overview", default=pick("title_body")[:2000])
    key_block = pick("key_insights", "insights", "key_insight")
    trends = pick("market_trends", "trends", "market_trend")
    comp = pick("competitor_analysis", "competitors", "competition")
    risks = pick("risks", "risk")
    opps = pick("opportunities", "opportunity")
    sources_block = pick("sources", "references", "bibliography")

    key_insights = _bullets_to_list(key_block) if key_block else _bullets_to_list(summary)
    if not key_insights:
        chunk = ((summary or markdown)[:800]).strip()
        key_insights = (
            [chunk] if len(chunk) > 3 else ["See full markdown report for narrative insights."]
        )

    sources = _urls_from_block(sources_block or markdown)
    if not sources:
        sources = _urls_from_block(markdown)

    summary_text = summary or "Summary not clearly sectioned; see full report."
    if len(summary_text) < 10:
        summary_text = (markdown[:400].strip() + "…") if markdown.strip() else "Summary unavailable."

    return RetailReport(
        title=title,
        summary=summary_text,
        key_insights=key_insights,
        market_trends=trends or "See analyst notes and sources.",
        competitor_analysis=comp or "See analyst notes and sources.",
        risks=risks or "See analyst notes and sources.",
        opportunities=opps or "See analyst notes and sources.",
        sources=sources if sources else ["No explicit URLs parsed; verify writer output."],
    )
