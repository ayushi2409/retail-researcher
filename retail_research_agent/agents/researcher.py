"""Research agent: discovers credible public sources via search tools."""

from __future__ import annotations

from typing import Any, List

from crewai import Agent
from crewai.tools import BaseTool


def create_researcher_agent(
    llm: Any,
    tools: List[BaseTool],
    *,
    verbose: bool = True,
    max_iter: int = 8,
) -> Agent:
    """Create the Research agent with web search tools attached."""
    return Agent(
        role="Retail Web Researcher",
        goal=(
            "Discover the most relevant, recent public sources for the user's retail question. "
            "Return 5–10 high-signal results with titles, URLs, and short snippets suitable "
            "for downstream scraping. Use few, sharp search queries—do not spam searches."
        ),
        backstory=(
            "You are an equity research associate specializing in consumer and retail. You "
            "hunt primary reporting, reputable trade press, government statistics, and "
            "recognized industry analysts. You are allergic to hallucinated links—you only cite "
            "URLs returned by your search tool. Prefer at most four distinct search queries; "
            "merge overlapping results mentally. When results are thin, broaden queries once "
            "and document uncertainty."
        ),
        tools=tools,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=max_iter,
    )
