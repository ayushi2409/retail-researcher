"""Centralized settings loaded from environment (Pydantic + dotenv)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the retail research agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (chat): OpenAI, Groq, Gemini, or local Ollama ---
    llm_provider: Literal["openai", "groq", "gemini", "ollama"] = Field(
        default="openai",
        description="Chat model backend for CrewAI agents",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (optional if using another LLM provider)",
    )
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key (free tier at console.groq.com)")
    google_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        description="Google AI Studio key for Gemini (free tier at aistudio.google.com)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL when llm_provider=ollama",
    )

    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Model when llm_provider=openai (or override via chat_model)",
    )
    chat_model: Optional[str] = Field(
        default=None,
        description="Override chat model for any provider; if unset, provider defaults apply",
    )

    # --- Embeddings: OpenAI, local Hugging Face (no key), or Ollama ---
    embedding_provider: Literal["auto", "openai", "huggingface", "ollama"] = Field(
        default="auto",
        description="auto: use OpenAI embeddings if OPENAI_API_KEY set, else HuggingFace local",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model when embedding_provider=openai",
    )
    huggingface_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local HF model (free; downloads on first use)",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model when embedding_provider=ollama",
    )

    tavily_api_key: Optional[str] = Field(default=None, description="Tavily search API key")
    serper_api_key: Optional[str] = Field(default=None, description="Google Serper API key")

    vector_backend: Literal["chroma", "faiss"] = Field(
        default="chroma",
        description="Vector store implementation for report embeddings",
    )
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="Directory for Chroma persistence",
    )
    faiss_index_dir: Path = Field(
        default=Path("./data/faiss"),
        description="Directory for persisted FAISS index",
    )
    reports_dir: Path = Field(default=Path("./data/reports"), description="Saved reports")
    cache_dir: Path = Field(default=Path("./data/cache"), description="Search/cache store")

    search_max_results: int = Field(default=8, ge=3, le=15)
    search_snippet_max_chars: int = Field(
        default=450,
        ge=120,
        le=2000,
        description="Max chars per search snippet in agent context (lower reduces LLM TPM usage)",
    )
    search_tool_response_max_chars: int = Field(
        default=4000,
        ge=600,
        le=80000,
        description="Hard cap on characters returned from retail_web_search each call (stops huge tool payloads)",
    )
    research_agent_max_iter: int = Field(
        default=8,
        ge=2,
        le=25,
        description="Max CrewAI iterations for the researcher agent (each may call tools)",
    )
    scrape_timeout_sec: int = Field(default=25, ge=5)
    scrape_max_urls: int = Field(default=12, ge=1, le=30)

    llm_num_retries: int = Field(
        default=6,
        ge=1,
        le=15,
        description="LiteLLM/CrewAI completion retries (helps with Groq 429 rate limits)",
    )
    llm_request_timeout_sec: int = Field(default=120, ge=30, le=600)

    request_max_retries: int = Field(default=3, ge=1, le=10)
    dedup_similarity_threshold: float = Field(default=0.88, ge=0.5, le=1.0)

    log_level: str = Field(default="INFO")
    enable_multi_hop: bool = Field(default=False, description="Second-pass search from gaps")
    enable_cache: bool = Field(default=True)
    enable_async_scrape: bool = Field(
        default=False,
        description="Use aiohttp + asyncio to fetch multiple URLs concurrently in scrape_urls",
    )

    @model_validator(mode="after")
    def _validate_llm_credentials(self) -> "Settings":
        p = self.llm_provider
        if p == "openai" and not (self.openai_api_key and self.openai_api_key.strip()):
            raise ValueError("Set OPENAI_API_KEY when LLM_PROVIDER=openai.")
        if p == "groq" and not (self.groq_api_key and self.groq_api_key.strip()):
            raise ValueError("Set GROQ_API_KEY when LLM_PROVIDER=groq (free: https://console.groq.com/).")
        if p == "gemini" and not (self.google_api_key and self.google_api_key.strip()):
            raise ValueError(
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY) when LLM_PROVIDER=gemini "
                "(free: https://aistudio.google.com/apikey)."
            )
        return self

    @model_validator(mode="after")
    def _validate_embedding_openai(self) -> "Settings":
        if self.embedding_provider == "openai" and not (self.openai_api_key and self.openai_api_key.strip()):
            raise ValueError("Set OPENAI_API_KEY when EMBEDDING_PROVIDER=openai.")
        return self

    @model_validator(mode="after")
    def _groq_context_limits(self) -> "Settings":
        """Groq free tier rejects large prompts (low TPM); tighten search output automatically."""
        if self.llm_provider != "groq":
            return self
        # mode='after' must return self; returning model_copy() is ignored by Pydantic v2.
        object.__setattr__(self, "search_max_results", min(self.search_max_results, 5))
        object.__setattr__(self, "search_snippet_max_chars", min(self.search_snippet_max_chars, 220))
        object.__setattr__(
            self,
            "search_tool_response_max_chars",
            min(self.search_tool_response_max_chars, 2200),
        )
        object.__setattr__(self, "research_agent_max_iter", min(self.research_agent_max_iter, 6))
        return self

    def resolved_chat_model(self) -> str:
        """Effective chat model name for the active LLM provider."""
        if self.chat_model and self.chat_model.strip():
            return self.chat_model.strip()
        defaults = {
            "openai": self.openai_model,
            # Lighter default: free-tier Groq TPM is easy to exceed with 70B + long tool output.
            "groq": "llama-3.1-8b-instant",
            "gemini": "gemini-2.0-flash",
            "ollama": "llama3.2",
        }
        return defaults[self.llm_provider]

    def resolved_embedding_provider(self) -> Literal["openai", "huggingface", "ollama"]:
        if self.embedding_provider == "auto":
            if self.openai_api_key and self.openai_api_key.strip():
                return "openai"
            return "huggingface"
        if self.embedding_provider == "openai":
            return "openai"
        if self.embedding_provider == "huggingface":
            return "huggingface"
        return "ollama"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
