#!/usr/bin/env python3
"""CLI entrypoint for the retail research agent."""

from __future__ import annotations

import argparse
import logging
import sys

if sys.version_info >= (3, 14):
    print(
        "Error: Python 3.14+ is not supported yet (CrewAI 1.x requires Python <3.14).\n"
        "Use Python 3.13:  python3.13 -m venv .venv\n"
        "See README.md (Setup) for Homebrew / install hints.",
        file=sys.stderr,
    )
    raise SystemExit(1)

import os

# Non-interactive defaults before CrewAI imports (avoids trace consent prompts on failed runs).
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from config.settings import get_settings
from crew import run_retail_research, similarity_lookup
from utils.logger import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-powered autonomous retail research agent")
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the latest retail trends in India in 2026?",
        help="Natural language research question",
    )
    parser.add_argument(
        "--vector-query",
        action="store_true",
        help="Instead of running the crew, run similarity search against stored reports",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Top-k matches for --vector-query",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.getLogger().setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    if args.vector_query:
        print(similarity_lookup(args.query, k=args.k))
        return 0

    result = run_retail_research(args.query, settings=settings)
    print("\n=== FINAL OUTPUT ===\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
