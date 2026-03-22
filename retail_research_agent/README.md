# Retail Research Agent

Production-oriented, modular **LLM + multi-agent** pipeline for autonomous **retail industry** research. It plans work, searches the web, scrapes pages, **deduplicates** evidence with **embeddings**, analyzes findings, writes a structured report, and persists **`.md` + `.txt`** plus a **vector index** (Chroma or FAISS).

## Stack

- Python 3.11+
- **CrewAI** (multi-agent orchestration)
- **LangChain** (embeddings + `Document`, Chroma/FAISS)
- **LLM**: OpenAI, **Groq**, **Google Gemini**, or **Ollama** (see [Free API keys](#free-api-keys-no-openai))
- **Embeddings**: OpenAI, **local Hugging Face** (no key), or **Ollama**
- **Tavily** or **Serper** (web search, with retries)
- **Chroma** or **FAISS** (local vector DB; `VECTOR_BACKEND`)
- **BeautifulSoup** + **lxml** (HTML cleaning), **aiohttp** (optional concurrent scraping)
- **Pydantic / pydantic-settings** (config + `schemas/report.py` validation)
- **tenacity** (retries)
- **python-dotenv** (via pydantic-settings)
- **pytest** (smoke tests)
- **Streamlit** (optional web UI — `streamlit_app.py`)

## Architecture

| Phase | Agents | What happens |
|-------|--------|----------------|
| 1 | Planner → Researcher → Scraper | Plan, `retail_web_search`, `retail_fetch_clean_text`, consolidated `### URL:` sections |
| — | (Python) | **Embedding-based deduplication** (`utils/corpus_dedup.py`) |
| 2 | Analyst → Writer → Storage | Reasoning, structured markdown report, `persist_retail_report` (disk + vectors) |

After the writer task, **Pydantic** validation parses the markdown into a `RetailReport` (`schemas/report.py`); failures are logged as warnings but do not stop the pipeline.

Advanced behaviors:

- **Source credibility** heuristics (`utils/helpers.py`)
- **Search caching** in `CACHE_DIR` (`ENABLE_CACHE`)
- **Multi-hop search** (`ENABLE_MULTI_HOP=true`)
- **Async multi-URL scraping** (`ENABLE_ASYNC_SCRAPE=true`)

## Free API keys (no OpenAI)

You can run the **agents** without OpenAI by using a free cloud key or a local model:

| Provider | Cost | Get a key / setup |
|----------|------|-------------------|
| **Groq** | Free tier | [console.groq.com](https://console.groq.com/) — create an API key |
| **Google Gemini** | Free tier | [Google AI Studio](https://aistudio.google.com/apikey) — create `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) |
| **Ollama** | Free (local) | Install from [ollama.com](https://ollama.com/), run `ollama pull llama3.2`, set `LLM_PROVIDER=ollama` |

**Embeddings** (vector DB + dedup): if you leave `OPENAI_API_KEY` unset, `EMBEDDING_PROVIDER=auto` uses **local Hugging Face** (`sentence-transformers/all-MiniLM-L6-v2`) — no API key; the model downloads on first use. If you *do* set `OPENAI_API_KEY`, auto switches to OpenAI embeddings unless you force `EMBEDDING_PROVIDER=huggingface`.

**Web search** still needs **Tavily** or **Serper** (or you get a placeholder message). Tavily/Serper often have trials or low-cost tiers; check their sites.

### Example `.env` (Groq + local embeddings)

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
# Optional: default Groq model is a fast 8B to stay under free-tier TPM; use 70B if you have headroom:
# CHAT_MODEL=llama-3.3-70b-versatile
# Do not set OPENAI_API_KEY if you want free local embeddings
EMBEDDING_PROVIDER=auto
TAVILY_API_KEY=tvly-...
```

**Groq rate limits:** the free **on-demand** tier enforces a low **TPM** budget. Groq may reject a single request if the **prompt is too large** (e.g. “Limit 6000, Requested 14000”)—that is usually **stacked tool output** (many `retail_web_search` calls), not just “requests per minute.” With `LLM_PROVIDER=groq`, the app **automatically** caps search breadth, snippet length, per-tool response size, and researcher iterations. You can still tune `SEARCH_MAX_RESULTS`, `SEARCH_SNIPPET_MAX_CHARS`, `SEARCH_TOOL_RESPONSE_MAX_CHARS`, and `RESEARCH_AGENT_MAX_ITER`. For heavier jobs, use **OpenAI/Gemini**, **Groq Dev tier**, or **Ollama** locally.

## Setup

**Python 3.11, 3.12, or 3.13 is required.** **Do not use Python 3.14** for this project yet: current **CrewAI 1.x** releases on PyPI declare `Requires-Python <3.14`, so `pip install -r requirements.txt` fails on 3.14 with “No matching distribution found for crewai”. Older CrewAI 0.11.x is incompatible with the LangChain 0.3 stack used here.

**Fix (macOS Homebrew):** install 3.13 and create the venv with it:

```bash
brew install python@3.13
cd retail_research_agent
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Fix (python.org):** install [Python 3.13](https://www.python.org/downloads/), then `python3.13 -m venv .venv` (adjust path if your installer uses a different command).

**tiktoken ≥0.12** helps 3.13 use prebuilt wheels instead of building Rust extensions.

```bash
cd retail_research_agent
python3.12 -m venv .venv   # or python3.11 / python3.13
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` uses **`crewai[litellm]`** so **Groq**, **Ollama**, and similar models work (CrewAI’s built-in providers are OpenAI, Gemini, etc.). If you ever see “LiteLLM fallback package is not installed”, run `pip install litellm` (or reinstall from `requirements.txt`).

Edit `.env`:

- **Chat**: either `OPENAI_API_KEY` with `LLM_PROVIDER=openai`, or `GROQ_API_KEY` / `GOOGLE_API_KEY` with `LLM_PROVIDER=groq` or `gemini`, or local **Ollama** (`LLM_PROVIDER=ollama`). Optional: `CHAT_MODEL` overrides the provider default; `LLM_NUM_RETRIES`, `LLM_REQUEST_TIMEOUT_SEC`, `SEARCH_SNIPPET_MAX_CHARS` tune resilience and context size.
- **Tracing**: `CREWAI_TRACING_ENABLED=false` (set by the CLI before import) keeps runs non-interactive; crews use `tracing=False` so first-run trace prompts do not block scripts.
- **Embeddings**: `EMBEDDING_PROVIDER=auto` (recommended) or `huggingface` / `openai` / `ollama`
- **Search**: `TAVILY_API_KEY` **or** `SERPER_API_KEY` for live web results
- Optional: `VECTOR_BACKEND=faiss`, `FAISS_INDEX_DIR`, `ENABLE_ASYNC_SCRAPE=true`

If you change embedding provider or model, delete the old vector folder (`data/chroma` or `data/faiss`) to avoid dimension mismatches.

## Run

```bash
cd retail_research_agent
source .venv/bin/activate
python main.py "What are the latest retail trends in India in 2026?"
```

Default query (if you omit the argument) is the India 2026 example above.

### Web UI (Streamlit)

From the same directory as `main.py` (so `.env` resolves correctly; the app also `chdir`s to its folder on startup):

```bash
cd retail_research_agent
source .venv/bin/activate
pip install -r requirements.txt   # includes streamlit
streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`). **New research** runs the full crew; **Search saved reports** calls the same vector similarity helper as `main.py --vector-query`.

### Query indexed reports (vector similarity)

After at least one successful run (embeds the query using your configured **embedding** backend — OpenAI, Hugging Face local, or Ollama):

```bash
python main.py "India grocery omnichannel" --vector-query --k 4
```

For a quick offline smoke check with an **empty FAISS** index (no embedding call), you can run:

```bash
VECTOR_BACKEND=faiss FAISS_INDEX_DIR=/tmp/empty_faiss_index python main.py "x" --vector-query --k 2
```

## Tests

[`pytest.ini`](pytest.ini) sets `pythonpath = .` so imports match the CLI layout.

```bash
cd retail_research_agent
source .venv/bin/activate
pytest tests/ -q
```

- Import and tool wiring smoke tests run without real API keys.
- **Live E2E** (optional): set `RUN_LIVE_E2E=1`, search keys (`TAVILY_API_KEY` or `SERPER_API_KEY`), and a real LLM (`OPENAI_API_KEY`, or `GROQ_API_KEY` + `LLM_PROVIDER=groq`, etc.), then `pytest tests/test_e2e.py -q`.

## Outputs

- **Reports**: `REPORTS_DIR` — each save produces matching **`.md` and `.txt`** files
- **Chroma** (default): `CHROMA_PERSIST_DIR`
- **FAISS** (when `VECTOR_BACKEND=faiss`): `FAISS_INDEX_DIR`
- **Search cache**: `CACHE_DIR`

## Project layout

```
retail_research_agent/
├── agents/           # CrewAI agent definitions
├── tools/            # Search, scrape, crew tools
├── memory/           # Vector store (Chroma / FAISS)
├── schemas/          # Pydantic report schema + markdown parser
├── utils/            # Logging, helpers, dedup
├── config/           # Pydantic settings
├── tests/            # pytest smoke + optional live E2E
├── crew.py           # Two-phase crew + aggregation + report validation
├── main.py           # CLI
├── streamlit_app.py  # Streamlit web UI
├── pytest.ini        # pythonpath for tests
├── requirements.txt
└── README.md
```

## Notes

- Without search API keys, the research agent still runs but receives a **configuration placeholder** instead of live results—add Tavily or Serper for real retrieval.
- Scraping respects `SCRAPE_MAX_URLS` and timeouts; failures are logged and surfaced as short scrape diagnostics in the corpus.
- **LangChain** powers embeddings and vector stores; **CrewAI** drives agents and tasks.

## License

Use and modify freely for your organization; add a license file if you redistribute.
