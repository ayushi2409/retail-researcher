"""Storage agent: persists the report to disk and vector memory."""

from __future__ import annotations

from typing import Any, List

from crewai import Agent
from crewai.tools import BaseTool


def create_storage_agent(
    llm: Any,
    tools: List[BaseTool],
    *,
    verbose: bool = True,
) -> Agent:
    """Create the Storage agent with persistence tools."""
    return Agent(
        role="Knowledge Repository Curator",
        goal=(
            "Take the final markdown report and persist it using persist_retail_report: "
            "save to local markdown and index embeddings for similarity search. Confirm paths "
            "and IDs returned by the tool."
        ),
        backstory=(
            "You are a meticulous research librarian for a retail insights platform. You "
            "ensure assets are stored once, with consistent naming, and that vector indexes "
            "stay aligned with on-disk artifacts. You do not alter report content during save."
        ),
        tools=tools,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=15,
    )
