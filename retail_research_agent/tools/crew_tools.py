"""CrewAI BaseTool implementations for search, scrape, dedup, and persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from memory.vector_store import ReportVectorStore
from tools.web_scraper import scrape_url
from tools.web_search import WebSearchService, format_search_results_for_prompt
class SearchQuery(BaseModel):
    query: str = Field(..., description="Focused retail-industry search query")


class UrlQuery(BaseModel):
    url: str = Field(..., description="Fully qualified http(s) URL to fetch")


class ReportSaveInput(BaseModel):
    title: str = Field(..., description="Short report title")
    body: str = Field(..., description="Full markdown report body")
    user_query: str = Field(..., description="Original user question")


def build_research_tools(settings: Optional[Settings] = None) -> List[BaseTool]:
    settings = settings or get_settings()
    service = WebSearchService(settings)

    class RetailWebSearchTool(BaseTool):
        name: str = "retail_web_search"
        description: str = (
            "Search the public web for retail industry facts, data, and news. "
            "Returns top sources with snippets and credibility hints. "
            "Do not invent URLs; only use URLs returned by this tool."
        )
        args_schema: Type[BaseModel] = SearchQuery

        def _run(self, query: str) -> str:
            results = service.search(query)
            return format_search_results_for_prompt(
                results,
                max_total_chars=settings.search_tool_response_max_chars,
            )

    return [RetailWebSearchTool()]


def build_scraper_tools(settings: Optional[Settings] = None) -> List[BaseTool]:
    settings = settings or get_settings()

    class RetailFetchTool(BaseTool):
        name: str = "retail_fetch_clean_text"
        description: str = (
            "Fetch a web page and return cleaned visible text (scripts/nav removed). "
            "Use only URLs returned from retail_web_search."
        )
        args_schema: Type[BaseModel] = UrlQuery

        def _run(self, url: str) -> str:
            return scrape_url(url, settings)

    return [RetailFetchTool()]


def build_storage_tools(
    vector_store: ReportVectorStore,
    settings: Optional[Settings] = None,
) -> List[BaseTool]:
    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    class PersistReportTool(BaseTool):
        name: str = "persist_retail_report"
        description: str = (
            "Save the final report to local disk as .md and .txt and add it to the vector "
            "database for later similarity search. Call once when the report is final."
        )
        args_schema: Type[BaseModel] = ReportSaveInput

        def _run(self, title: str, body: str, user_query: str) -> str:
            safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", title)[:80] or "report"
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            base = Path(settings.reports_dir) / f"{ts}_{safe}"
            md_path = base.with_suffix(".md")
            txt_path = base.with_suffix(".txt")
            md_content = f"# {title}\n\n{body}"
            md_path.write_text(md_content, encoding="utf-8")
            txt_path.write_text(md_content, encoding="utf-8")
            report_id = vector_store.add_report(
                body,
                user_query=user_query,
                title=title,
                extra_metadata={"path_md": str(md_path), "path_txt": str(txt_path)},
            )
            return (
                f"Saved markdown to {md_path}, text copy to {txt_path}, "
                f"indexed as report_id={report_id}"
            )

    return [PersistReportTool()]
