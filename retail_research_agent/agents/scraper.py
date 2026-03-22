"""Scraper agent: pulls clean text from URLs found during research."""

from __future__ import annotations

from typing import Any, List

from crewai import Agent
from crewai.tools import BaseTool


def create_scraper_agent(
    llm: Any,
    tools: List[BaseTool],
    *,
    verbose: bool = True,
) -> Agent:
    """Create the Scraper agent with HTML-to-text tooling."""
    return Agent(
        role="Retail Content Extractor",
        goal=(
            "Fetch selected URLs from the research step and extract readable article text. "
            "Emit a single consolidated document where each section begins with "
            "'### URL: <url>' followed by the cleaned body text."
        ),
        backstory=(
            "You are a senior data engineer who has built news pipelines for quant funds. "
            "You strip navigation, ads, and boilerplate, preserve factual paragraphs, and "
            "skip paywalled or failed pages gracefully. You never guess page content—only "
            "what the fetch tool returns."
        ),
        tools=tools,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=35,
    )
