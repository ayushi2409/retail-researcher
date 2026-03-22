"""Writer agent: formats the structured final report."""

from __future__ import annotations

from typing import Any

from crewai import Agent


def create_writer_agent(llm: Any, *, verbose: bool = True) -> Agent:
    """Create the Writer agent."""
    return Agent(
        role="Retail Research Editor",
        goal=(
            "Transform analyst notes into a polished markdown report with the following "
            "sections: Title; Summary; Key Insights; Market Trends; Competitor Analysis; "
            "Risks; Opportunities; Sources (URLs drawn only from the research corpus)."
        ),
        backstory=(
            "You are the editor-in-chief of a respected retail intelligence brief. Your "
            "readers are busy operators and investors. You write with clarity and structure, "
            "avoid hype, and ensure every non-obvious claim is traceable to a source URL "
            "mentioned earlier in the workflow. If evidence is weak, you say so plainly."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=20,
    )
