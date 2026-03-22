"""Streamlit web UI for the retail research agent."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st

if sys.version_info >= (3, 14):
    st.error(
        "Python 3.14+ is not supported yet (CrewAI 1.x requires Python <3.14). "
        "Use Python 3.13 and reinstall dependencies. See README.md."
    )
    st.stop()

_APP_ROOT = Path(__file__).resolve().parent
os.chdir(_APP_ROOT)
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from config.settings import get_settings  # noqa: E402
from crew import run_retail_research, similarity_lookup  # noqa: E402

st.set_page_config(
    page_title="Retail Research Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_log_level() -> None:
    settings = get_settings()
    logging.getLogger().setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


def _sidebar() -> None:
    settings = get_settings()
    st.sidebar.title("Retail Research Agent")
    st.sidebar.markdown(
        "Multi-agent pipeline: plan → search → scrape → analyze → report → save to disk + vector index."
    )
    st.sidebar.divider()
    st.sidebar.caption("Runtime (from `.env`)")
    st.sidebar.markdown(
        f"- **LLM provider:** `{settings.llm_provider}`\n"
        f"- **Chat model:** `{settings.resolved_chat_model()}`\n"
        f"- **Embeddings:** `{settings.resolved_embedding_provider()}`\n"
        f"- **Vector store:** `{settings.vector_backend}`"
    )
    st.sidebar.caption(f"App directory: `{_APP_ROOT}`")


def main() -> None:
    _apply_log_level()
    _sidebar()

    if "last_research_result" not in st.session_state:
        st.session_state.last_research_result = None
    if "last_research_query" not in st.session_state:
        st.session_state.last_research_query = None

    tab_run, tab_search = st.tabs(["New research", "Search saved reports"])

    with tab_run:
        st.subheader("Run a research job")
        query = st.text_area(
            "Research question",
            value="What are the latest retail trends in India in 2026?",
            height=100,
            help="Natural-language question for the crew (planner, researcher, scraper, analyst, writer).",
        )
        run = st.button("Run research", type="primary", use_container_width=True)

        if run:
            q = (query or "").strip()
            if not q:
                st.warning("Enter a question first.")
            else:
                st.session_state.last_research_result = None
                st.session_state.last_research_query = q
                with st.status("Running pipeline (this may take several minutes)…", expanded=True) as status:
                    try:
                        settings = get_settings()
                        result = run_retail_research(q, settings=settings)
                        status.update(label="Done", state="complete")
                    except Exception as exc:
                        status.update(label="Failed", state="error")
                        st.error(f"**{type(exc).__name__}:** {exc}")
                        st.exception(exc)
                    else:
                        st.session_state.last_research_result = result
                        st.success("Pipeline finished.")

        if st.session_state.last_research_result is not None:
            st.markdown("### Final output")
            if st.session_state.last_research_query:
                st.caption(f"Query: {st.session_state.last_research_query}")
            st.markdown(st.session_state.last_research_result)

    with tab_search:
        st.subheader("Similarity search over indexed reports")
        st.caption("Uses the same vector store as the CLI (`python main.py --vector-query`).")
        sq = st.text_input("Query past reports", placeholder="e.g. quick commerce India")
        k = st.slider("Top-k matches", min_value=1, max_value=12, value=4)
        if st.button("Search index", use_container_width=True):
            t = (sq or "").strip()
            if not t:
                st.warning("Enter a search phrase.")
            else:
                try:
                    out = similarity_lookup(t, k=k)
                    st.markdown(out)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"**{type(exc).__name__}:** {exc}")
                    st.exception(exc)


if __name__ == "__main__":
    main()
