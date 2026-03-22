"""Planner agent: decomposes the user query into an execution strategy."""

from __future__ import annotations

from typing import Any

from crewai import Agent


def create_planner_agent(llm: Any, *, verbose: bool = True) -> Agent:
    """
    Create the Planner agent.

    Breaks the retail research question into sub-tasks and defines sequencing hints
    for search, scraping, and synthesis.
    """
    return Agent(
        role="Retail Research Strategist",
        goal=(
            "Translate the user's retail research question into a concise, actionable plan: "
            "sub-questions, target geographies/segments, required data types, and a clear "
            "execution order for web research and analysis."
        ),
        backstory=(
            "You are a principal strategy consultant focused on global retail. You have led "
            "dozens of market studies for grocers, apparel, and e-commerce platforms. You "
            "excel at scoping ambiguous questions, avoiding scope creep, and specifying what "
            "evidence would change a leadership decision. You never fabricate facts—plans "
            "reference the types of sources to seek, not invented statistics."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=20,
    )
