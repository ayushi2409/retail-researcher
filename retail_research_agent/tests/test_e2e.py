"""
Live end-to-end run (optional).

Set ``RUN_LIVE_E2E=1`` plus search keys (``TAVILY_API_KEY`` or ``SERPER_API_KEY``) and a real LLM:

- ``LLM_PROVIDER=openai`` + ``OPENAI_API_KEY``, or
- ``LLM_PROVIDER=groq`` + ``GROQ_API_KEY``, or
- ``LLM_PROVIDER=gemini`` + ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``, or
- ``LLM_PROVIDER=ollama`` (local Ollama running)

Otherwise this test is skipped.
"""

from __future__ import annotations

import os

import pytest


def _live_keys_configured() -> bool:
    if os.getenv("RUN_LIVE_E2E") != "1":
        return False
    if not (os.getenv("TAVILY_API_KEY") or os.getenv("SERPER_API_KEY")):
        return False
    provider = (os.getenv("LLM_PROVIDER") or "openai").lower()
    if provider == "openai":
        oa = os.getenv("OPENAI_API_KEY", "")
        return bool(oa and not oa.startswith("sk-test-placeholder"))
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY", "").strip())
    if provider == "gemini":
        return bool((os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip())
    if provider == "ollama":
        return True
    return False


@pytest.mark.skipif(not _live_keys_configured(), reason="Set RUN_LIVE_E2E=1 and real API keys")
def test_live_retail_research_short_query() -> None:
    from crew import run_retail_research

    out = run_retail_research("What is omnichannel retail? One paragraph answer scope.")
    assert out
    assert len(str(out)) > 20
