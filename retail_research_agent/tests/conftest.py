"""Pytest fixtures: minimal env so Settings() loads without a real .env file."""

from __future__ import annotations

import os

import pytest

from config.settings import get_settings


def pytest_configure() -> None:
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-not-for-production")
    os.environ.setdefault("VECTOR_BACKEND", "chroma")


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    yield
    get_settings.cache_clear()
