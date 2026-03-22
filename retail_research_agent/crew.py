"""CrewAI orchestration: research phase, embedding dedup, analysis & persistence."""

from __future__ import annotations

from typing import Any, Optional

from crewai import Crew, Process, Task
from pydantic import ValidationError

from agents.analyst import create_analyst_agent
from agents.planner import create_planner_agent
from agents.researcher import create_researcher_agent
from agents.scraper import create_scraper_agent
from agents.storage import create_storage_agent
from agents.writer import create_writer_agent
from config.settings import Settings, get_settings
from memory.vector_store import ReportVectorStore
from tools.crew_tools import (
    build_research_tools,
    build_scraper_tools,
    build_storage_tools,
)
from schemas.report import parse_and_validate_retail_report
from utils.corpus_dedup import deduplicate_scraped_corpus
from utils.logger import get_logger, log_step

logger = get_logger(__name__)


def _get_task_output(task: Any) -> str:
    """Normalize CrewAI task output to string across versions."""
    out = getattr(task, "output", None)
    if out is None:
        return ""
    raw = getattr(out, "raw", None)
    if raw is not None:
        return str(raw)
    return str(out)


def _build_llm(settings: Settings) -> Any:
    """
    Instantiate CrewAI LLM.

    Supports OpenAI, Groq, Gemini (via LiteLLM), and local Ollama. See README for free keys.
    """
    from crewai import LLM

    model = settings.resolved_chat_model()
    provider = settings.llm_provider
    common = {
        "timeout": settings.llm_request_timeout_sec,
        "num_retries": settings.llm_num_retries,
    }

    if provider == "openai":
        return LLM(model=model, api_key=settings.openai_api_key, **common)

    if provider == "groq":
        return LLM(model=f"groq/{model}", api_key=settings.groq_api_key, **common)

    if provider == "gemini":
        return LLM(model=f"gemini/{model}", api_key=settings.google_api_key, **common)

    if provider == "ollama":
        return LLM(
            model=f"ollama/{model}",
            base_url=settings.ollama_base_url,
            api_key="ollama",
            **common,
        )

    raise ValueError(f"Unsupported llm_provider: {provider}")


def run_retail_research(
    user_query: str,
    *,
    settings: Optional[Settings] = None,
) -> str:
    """
    Execute the full multi-agent pipeline and return the final stored confirmation / report.

    Flow: plan → search → scrape → programmatic dedup → analyze → write → persist.
    """
    settings = settings or get_settings()
    log_step(logger, "pipeline_start", details={"query_preview": user_query[:160]})

    try:
        from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages

        set_suppress_tracing_messages(True)
    except Exception:
        pass

    llm = _build_llm(settings)
    vector_store = ReportVectorStore(settings)

    planner = create_planner_agent(llm)
    researcher = create_researcher_agent(
        llm,
        build_research_tools(settings),
        max_iter=settings.research_agent_max_iter,
    )
    scraper = create_scraper_agent(llm, build_scraper_tools(settings))

    multi_hop_clause = ""
    if settings.enable_multi_hop:
        multi_hop_clause = (
            "\nAfter your first search round, run ONE additional focused query to fill "
            "evident gaps (e.g., geography, channel, or time horizon) before finishing.\n"
        )

    task_plan = Task(
        description=(
            "User question: {user_query}\n\n"
            "Produce a concise research execution plan: bullet sub-questions, suggested "
            "search angles, geographies/segments, and what 'good enough' evidence looks like."
        ),
        expected_output=(
            "Markdown plan with numbered sub-tasks and explicit search keywords/phrases "
            "the research agent should use."
        ),
        agent=planner,
    )

    task_research = Task(
        description=(
            "User question: {user_query}\n\n"
            "Follow the planner output in context. Use retail_web_search with tight queries "
            "(at most four separate tool calls; avoid near-duplicate queries). "
            "Return 5–10 distinct sources with title, URL, snippet, and credibility cues."
            f"{multi_hop_clause}"
        ),
        expected_output=(
            "Markdown list of sources. Each item MUST include a real URL from the tool output."
        ),
        agent=researcher,
        context=[task_plan],
    )

    task_scrape = Task(
        description=(
            "User question: {user_query}\n\n"
            "Using ONLY URLs from the research task, call retail_fetch_clean_text for each "
            "distinct URL (cap at a reasonable number). Merge into one document where every "
            "section starts with a line exactly like:\n"
            "### URL: https://...\n"
            "followed by cleaned text. Skip unusable pages but mention the skip briefly."
        ),
        expected_output="Single markdown document with ### URL headers and extracted text.",
        agent=scraper,
        context=[task_research],
    )

    research_crew = Crew(
        agents=[planner, researcher, scraper],
        tasks=[task_plan, task_research, task_scrape],
        process=Process.sequential,
        verbose=True,
        tracing=False,
    )

    research_crew.kickoff(inputs={"user_query": user_query})
    scraped = _get_task_output(task_scrape)
    log_step(
        logger,
        "phase_research_done",
        details={"scrape_chars": len(scraped)},
    )

    log_step(logger, "aggregation_start")
    deduped = deduplicate_scraped_corpus(scraped, settings)
    max_ctx = 80_000
    corpus_for_llm = deduped if len(deduped) <= max_ctx else deduped[:max_ctx] + "\n\n[TRUNCATED]\n"
    log_step(
        logger,
        "aggregation_done",
        details={"dedup_chars": len(deduped), "llm_chars": len(corpus_for_llm)},
    )

    analyst = create_analyst_agent(llm)
    writer = create_writer_agent(llm)
    storage = create_storage_agent(llm, build_storage_tools(vector_store, settings))

    task_analyze = Task(
        description=(
            f"User question:\n{user_query}\n\n"
            f"Deduplicated research corpus:\n{corpus_for_llm}\n\n"
            "Deliver deep analysis: trends, insights, risks, opportunities. "
            "Cite which URLs support each major point. If evidence is weak, say so."
        ),
        expected_output=(
            "Structured analyst markdown with sections: Evidence Overview, Trends, "
            "Insights, Risks, Opportunities, Conflicts/Gaps, Source map (URL → claim)."
        ),
        agent=analyst,
    )

    task_write = Task(
        description=(
            "Using the analyst output in context, write the final client-facing markdown report. "
            "Sections required: Title; Summary; Key Insights; Market Trends; "
            "Competitor Analysis; Risks; Opportunities; Sources (bulleted URLs from corpus)."
        ),
        expected_output="Polished markdown report matching the required headings.",
        agent=writer,
        context=[task_analyze],
    )

    task_store = Task(
        description=(
            "Persist the final report. Extract title and body from the writer output, then call "
            "persist_retail_report(title, body, user_query) exactly once with:\n"
            f"user_query={user_query!r}\n"
            "Return the tool's confirmation string as your final answer."
        ),
        expected_output="Confirmation of file path and vector index id from the tool.",
        agent=storage,
        context=[task_write],
    )

    analysis_crew = Crew(
        agents=[analyst, writer, storage],
        tasks=[task_analyze, task_write, task_store],
        process=Process.sequential,
        verbose=True,
        tracing=False,
    )

    final = analysis_crew.kickoff()
    writer_md = _get_task_output(task_write)
    try:
        parse_and_validate_retail_report(writer_md)
        log_step(logger, "report_validation", status="ok")
    except ValidationError as exc:
        log_step(
            logger,
            "report_validation",
            status="warn",
            details={"errors": exc.errors()[:8]},
        )

    log_step(logger, "pipeline_complete")
    return str(final)


def similarity_lookup(query: str, k: int = 4) -> str:
    """Optional helper: query the vector store after reports are indexed."""
    settings = get_settings()
    store = ReportVectorStore(settings)
    docs = store.similarity_search(query, k=k)
    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"### Match {i}\n{d.page_content[:2000]}\nMetadata: {d.metadata}\n")
    return "\n".join(parts) if parts else "No indexed reports yet."
