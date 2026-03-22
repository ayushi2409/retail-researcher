"""Smoke imports and CrewAI BaseTool wiring."""

from __future__ import annotations

import importlib


def test_import_crew_and_agents() -> None:
    importlib.import_module("crew")
    importlib.import_module("agents.planner")
    importlib.import_module("tools.crew_tools")
    importlib.import_module("memory.vector_store")
    importlib.import_module("schemas.report")


def test_build_tools_instantiate() -> None:
    from tools.crew_tools import build_research_tools, build_scraper_tools, build_storage_tools
    from memory.vector_store import ReportVectorStore

    rs = build_research_tools()
    ss = build_scraper_tools()
    vs = ReportVectorStore()
    st = build_storage_tools(vs)
    assert rs and hasattr(rs[0], "name")
    assert ss and hasattr(ss[0], "name")
    assert st and hasattr(st[0], "name")
