"""Analyst agent: deep reasoning over deduplicated evidence."""

from __future__ import annotations

from typing import Any

from crewai import Agent


def create_analyst_agent(llm: Any, *, verbose: bool = True) -> Agent:
    """Create the Analyst agent (no tools; works on provided corpus)."""
    return Agent(
        role="Retail Industry Analyst",
        goal=(
            "Perform rigorous analysis on the provided research corpus: identify trends, "
            "insights, risks, and opportunities grounded strictly in the supplied text. "
            "Flag conflicts between sources and avoid extrapolation beyond evidence."
        ),
        backstory=(
            "You are a director-level retail analyst at a top-tier strategy firm. You "
            "synthesize messy field notes into executive-ready implications. You cite which "
            "themes are well-supported versus speculative. You are explicit when data is "
            "stale, regional, or incomplete, and you separate facts from interpretation."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=25,
    )
