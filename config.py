"""
Central configuration for the Indian AI Portfolio Pipeline.
All settings are controlled from here — edit this file to tune behavior.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
PIPELINE_RUNS_DIR = STORAGE_DIR / "pipeline_runs"
PORTFOLIO_FILE = STORAGE_DIR / "portfolio.json"
TRADES_LOG_FILE = STORAGE_DIR / "trades_log.json"
THESIS_ALERTS_FILE = STORAGE_DIR / "thesis_alerts.json"
DB_FILE = STORAGE_DIR / "portfolio.sqlite3"
CACHE_DIR = BASE_DIR / ".cache"
SOURCE_PAYLOADS_DIR = STORAGE_DIR / "source_payloads"

# Ensure directories exist
for d in [STORAGE_DIR, PIPELINE_RUNS_DIR, CACHE_DIR, SOURCE_PAYLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Claude API
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")                               # HuggingFace token — optional, raises FinBERT rate limit

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "")
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "").strip()
OPENAI_COMPAT_EXTRA_BODY_JSON = os.getenv("OPENAI_COMPAT_EXTRA_BODY_JSON", "").strip()
OPENAI_COMPAT_DISABLE_QWEN_THINKING = os.getenv("OPENAI_COMPAT_DISABLE_QWEN_THINKING", "true").lower() in ("1", "true", "yes")

# Provider retry controls (Gemini/Groq HTTP)
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "1.0"))
LLM_RETRY_MAX_SECONDS = float(os.getenv("LLM_RETRY_MAX_SECONDS", "20.0"))
LLM_HTTP_TIMEOUT_SECONDS = int(os.getenv("LLM_HTTP_TIMEOUT_SECONDS", "300"))

# Budget guardrails (soft, best-effort enforcement where implemented)
MAX_MONTHLY_BUDGET_USD = float(os.getenv("MAX_MONTHLY_BUDGET_USD", "0") or "0")  # 0 disables

# Optional non-Anthropic pricing (per 1M tokens) for cost estimation only.
# Defaults to 0.0 because providers frequently change pricing and may be billed by character/compute tier.
GEMINI_INPUT_PER_M = float(os.getenv("GEMINI_INPUT_PER_M", "0") or "0")
GEMINI_OUTPUT_PER_M = float(os.getenv("GEMINI_OUTPUT_PER_M", "0") or "0")
GROQ_INPUT_PER_M = float(os.getenv("GROQ_INPUT_PER_M", "0") or "0")
GROQ_OUTPUT_PER_M = float(os.getenv("GROQ_OUTPUT_PER_M", "0") or "0")
MISTRAL_INPUT_PER_M = float(os.getenv("MISTRAL_INPUT_PER_M", "0") or "0")
MISTRAL_OUTPUT_PER_M = float(os.getenv("MISTRAL_OUTPUT_PER_M", "0") or "0")

# Model tiers — swap these to change quality/cost
# You can override via env, e.g.:
#   MODEL_FAST="gemini:gemma-4-31b-it"
#   MODEL_FAST="groq:llama-3.1-8b-instant"
#   MODEL_FAST="openai:Qwen/Qwen2.5-7B-Instruct"
#   OPENAI_COMPAT_MODEL="Qwen/Qwen2.5-7B-Instruct"  # routes all default tiers through OPENAI_COMPAT_BASE_URL
_DEFAULT_MODEL_FAST = "gemini:gemma-4-31b-it"
_DEFAULT_MODEL_SMART = "gemini:gemma-4-31b-it"
_DEFAULT_MODEL_BEST = "gemini:gemma-4-31b-it"
_OPENAI_COMPAT_MODEL_SPEC = (
    OPENAI_COMPAT_MODEL
    if not OPENAI_COMPAT_MODEL or ":" in OPENAI_COMPAT_MODEL
    else f"openai:{OPENAI_COMPAT_MODEL}"
)


import logging as _logging
_model_env_logger = _logging.getLogger(__name__)


def _model_env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    # Only override when the user did NOT set the env var explicitly.
    # An explicit STAGE*_MODEL/MODEL_* value is always respected, even if it
    # matches an Anthropic default — silent downgrades caused stage-quality bugs.
    if _OPENAI_COMPAT_MODEL_SPEC and not value:
        _model_env_logger.warning(
            "Model %s unset; routing to OPENAI_COMPAT_MODEL=%s (default would have been %s).",
            name, _OPENAI_COMPAT_MODEL_SPEC, default,
        )
        return _OPENAI_COMPAT_MODEL_SPEC
    return value or default


MODEL_FAST = _model_env("MODEL_FAST", _DEFAULT_MODEL_FAST)          # Stage 1 bulk screening (cheap)
MODEL_SMART = _model_env("MODEL_SMART", _DEFAULT_MODEL_SMART)      # Stages 2–5 (quality + speed)
MODEL_BEST = _model_env("MODEL_BEST", _DEFAULT_MODEL_BEST)          # Stage 4 portfolio construction (premium)

# Default models per stage
STAGE1_MODEL = _model_env("STAGE1_MODEL", MODEL_FAST)
STAGE2_MODEL = _model_env("STAGE2_MODEL", MODEL_SMART)
STAGE3_MODEL = _model_env("STAGE3_MODEL", MODEL_SMART)
STAGE4_MODEL = _model_env("STAGE4_MODEL", MODEL_SMART)     # Upgrade to MODEL_BEST for highest quality
STAGE5_MODEL = _model_env("STAGE5_MODEL", MODEL_SMART)

# ─────────────────────────────────────────────
# Token Optimization (Caveman & Compression)
# ─────────────────────────────────────────────
CAVEMAN_MODE = True            # Forces terse "caveman" logic to save 75% output tokens
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "0"))
TEMPERATURE = 0.2              # Low for analytical tasks; higher = more creative

# ─────────────────────────────────────────────
# Universe & Benchmark
# ─────────────────────────────────────────────
NIFTY500_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
NIFTY500_MIRROR_URLS = [
    # Live mirrors used when niftyindices.com blocks automation (HTTP 403).
    "https://raw.githubusercontent.com/Hpareek07/NSEData/master/ind_nifty500list.csv",
    "https://raw.githubusercontent.com/chaitanyarahalkar/Financial-Info-Extractor/master/ind_nifty500list.csv",
    "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv",
    "https://raw.githubusercontent.com/Ayush21031/Finance/main/ind_nifty500list.csv",
    "https://raw.githubusercontent.com/AnshSavarkar/Stock-Market-Prediction/main/ind_nifty500list.csv",
    "https://raw.githubusercontent.com/dswh/NER_News_Feed/master/data/ind_nifty500list.csv",
    "https://raw.githubusercontent.com/HimanshuKhandal/NSE-Stock-Data/master/ind_nifty500list.csv",
]
REQUIRE_FULL_UNIVERSE = os.getenv("REQUIRE_FULL_UNIVERSE", "true").lower() in ("1", "true", "yes")
MIN_UNIVERSE_SIZE = int(os.getenv("MIN_UNIVERSE_SIZE", "500"))
MIN_FUNDAMENTALS_COVERAGE_PCT = float(os.getenv("MIN_FUNDAMENTALS_COVERAGE_PCT", "0.95"))
BENCHMARK_SYMBOL = "^NSEI"                # Nifty 50 index on Yahoo Finance
BENCHMARK_NAME = "Nifty 50"
BENCHMARK_ANNUAL_RETURN_TARGET = 0.12     # 12% — portfolio must exceed this in model

# ─────────────────────────────────────────────
# Paper Portfolio
# ─────────────────────────────────────────────
PAPER_PORTFOLIO_VALUE_INR = 1_000_000     # ₹10,00,000 (10 Lakh) starting capital

# ─────────────────────────────────────────────
# Portfolio Hard Constraints
# ─────────────────────────────────────────────
MAX_POSITIONS = 15
MIN_POSITIONS = 10
MAX_POSITION_PCT = 0.15                   # Max 15% in any single stock
MIN_POSITION_PCT = 0.03                   # Min 3% per position
MAX_SECTOR_PCT = 0.35                     # Max 35% in any one sector
MIN_EXPECTED_RETURN = 0.0                 # All positions must have positive EV

# ─────────────────────────────────────────────
# Pipeline Schedule
# ─────────────────────────────────────────────
REBALANCE_EVERY_DAYS = 3
THESIS_MONITOR_HOUR = 6                   # Run thesis monitor at 6 AM IST daily

# Autonomous daemon scheduler (timezone-aware)
SCHEDULER_TIMEZONE = os.getenv("SCHEDULER_TIMEZONE", "Asia/Kolkata")
DAILY_SOURCE_SYNC_TIME = os.getenv("DAILY_SOURCE_SYNC_TIME", "05:15")  # HH:MM in SCHEDULER_TIMEZONE
DAILY_PIPELINE_TIME = os.getenv("DAILY_PIPELINE_TIME", "21:30")  # HH:MM in SCHEDULER_TIMEZONE
DAILY_MONITOR_TIME = os.getenv("DAILY_MONITOR_TIME", "06:00")    # HH:MM in SCHEDULER_TIMEZONE
PIPELINE_DAYS = os.getenv("PIPELINE_DAYS", "")  # e.g. "MON,WED" or empty for daily
OFFCYCLE_REBALANCE_ON_URGENT = os.getenv("OFFCYCLE_REBALANCE_ON_URGENT", "true").lower() in ("1", "true", "yes")
OFFCYCLE_MIN_HOURS_BETWEEN_PIPELINES = float(os.getenv("OFFCYCLE_MIN_HOURS_BETWEEN_PIPELINES", "6"))
DAILY_SOURCE_SYNC_ENABLED = os.getenv("DAILY_SOURCE_SYNC_ENABLED", "true").lower() in ("1", "true", "yes")
DAILY_ETL_ENABLED = os.getenv("DAILY_ETL_ENABLED", "true").lower() in ("1", "true", "yes")
DAILY_BS_RESEARCH_SYNC_ENABLED = os.getenv("DAILY_BS_RESEARCH_SYNC_ENABLED", "true").lower() in ("1", "true", "yes")
BS_RESEARCH_SYNC_DAYS = int(os.getenv("BS_RESEARCH_SYNC_DAYS", "30"))
BS_RESEARCH_SYNC_MAX_PDFS = int(os.getenv("BS_RESEARCH_SYNC_MAX_PDFS", "20"))
BS_RESEARCH_SYNC_HEADLESS = os.getenv("BS_RESEARCH_SYNC_HEADLESS", "true").lower() in ("1", "true", "yes")
BS_RESEARCH_SYNC_SLOW_MO_MS = int(os.getenv("BS_RESEARCH_SYNC_SLOW_MO_MS", "0"))
BS_RESEARCH_SYNC_WAIT_MS = int(os.getenv("BS_RESEARCH_SYNC_WAIT_MS", "3000"))
BS_RESEARCH_SYNC_SCROLLS = int(os.getenv("BS_RESEARCH_SYNC_SCROLLS", "5"))
BS_RESEARCH_USER_DATA_DIR = os.getenv("BS_RESEARCH_USER_DATA_DIR", str(STORAGE_DIR / "bs_profile"))
BS_RESEARCH_OUT_DIR = os.getenv("BS_RESEARCH_OUT_DIR", str(STORAGE_DIR / "bs_research"))

# Daily-lite mode (Mon–Thu by default in daemon; optional CLI mode)
DAILY_LITE_NEW_CANDIDATES = int(os.getenv("DAILY_LITE_NEW_CANDIDATES", "3"))
DAILY_LITE_FLAGGED_HOLDINGS = int(os.getenv("DAILY_LITE_FLAGGED_HOLDINGS", "2"))
DAILY_LITE_S2_MAX_WORKERS = int(os.getenv("DAILY_LITE_S2_MAX_WORKERS", "2"))
DAILY_LITE_S3_MAX_WORKERS = int(os.getenv("DAILY_LITE_S3_MAX_WORKERS", "2"))

# ─────────────────────────────────────────────
# Stage 1 Screening
# ─────────────────────────────────────────────
STAGE1_TOP_N = int(os.getenv("STAGE1_TOP_N", "50"))                         # Top 50 advance from Stage 1
SENTIMENT_WEIGHT = float(os.getenv("SENTIMENT_WEIGHT", "0.08"))
PIPELINE_STRICT_VALIDATION = os.getenv("PIPELINE_STRICT_VALIDATION", "true").lower() in ("1", "true", "yes")
PRODUCTION_MODE = os.getenv("PRODUCTION_MODE", "false").lower() in ("1", "true", "yes")  # fail-closed on data errors when True             # Weight of FinBERT sentiment in Stage 1 composite (0 = off)
STAGE2_TOP_N = int(os.getenv("STAGE2_TOP_N", "30"))                         # Adversarial research run on top N
_stage1_use_llm_raw = os.getenv("STAGE1_USE_LLM_REVIEW", "").strip().lower()
STAGE1_USE_LLM_REVIEW = True if _stage1_use_llm_raw == "" else _stage1_use_llm_raw in ("1", "true", "yes")
STAGE2_BULL_AGENTS_PER_STOCK = int(os.getenv("STAGE2_BULL_AGENTS_PER_STOCK", "2"))
STAGE2_BEAR_AGENTS_PER_STOCK = int(os.getenv("STAGE2_BEAR_AGENTS_PER_STOCK", "2"))
STAGE2_RESEARCH_LOOKBACK_DAYS = int(os.getenv("STAGE2_RESEARCH_LOOKBACK_DAYS", "7"))
PIPELINE_S2_MAX_WORKERS = int(os.getenv("PIPELINE_S2_MAX_WORKERS", "8"))
PIPELINE_S3_MAX_WORKERS = int(os.getenv("PIPELINE_S3_MAX_WORKERS", "8"))
STAGE2_PARALLEL_ACROSS_STOCKS = os.getenv("STAGE2_PARALLEL_ACROSS_STOCKS", "true").lower() in ("1", "true", "yes")
STAGE2_SYNTHESIS_MAX_WORKERS = int(os.getenv("STAGE2_SYNTHESIS_MAX_WORKERS", str(PIPELINE_S2_MAX_WORKERS)))
STAGE2_BETWEEN_SIDES_SLEEP_SECONDS = float(os.getenv("STAGE2_BETWEEN_SIDES_SLEEP_SECONDS", "0"))
STAGE2_BETWEEN_STOCKS_SLEEP_SECONDS = float(os.getenv("STAGE2_BETWEEN_STOCKS_SLEEP_SECONDS", "0"))
STAGE2_AGENT_MAX_TOKENS = int(os.getenv("STAGE2_AGENT_MAX_TOKENS", str(MAX_TOKENS)))
STAGE3_AGENT_MAX_TOKENS = int(os.getenv("STAGE3_AGENT_MAX_TOKENS", str(MAX_TOKENS)))
STAGE4_AGENT_MAX_TOKENS = int(os.getenv("STAGE4_AGENT_MAX_TOKENS", str(MAX_TOKENS)))
STAGE5_AGENT_MAX_TOKENS = int(os.getenv("STAGE5_AGENT_MAX_TOKENS", str(MAX_TOKENS)))
SYNTHESIS_AGENT_MAX_TOKENS = int(os.getenv("SYNTHESIS_AGENT_MAX_TOKENS", str(MAX_TOKENS)))
SYNTHESIS_VALIDATOR_MAX_TOKENS = int(os.getenv("SYNTHESIS_VALIDATOR_MAX_TOKENS", str(MAX_TOKENS)))
STAGE4_DETERMINISTIC_OPTIMIZER = os.getenv("STAGE4_DETERMINISTIC_OPTIMIZER", "true").lower() in ("1", "true", "yes")
STAGE4_WEIGHT_TEMPERATURE = float(os.getenv("STAGE4_WEIGHT_TEMPERATURE", "0.03"))      # Softmax sharpness; lower = winner-takes-more
STAGE4_EV_PENALTY_THRESHOLD = float(os.getenv("STAGE4_EV_PENALTY_THRESHOLD", "0.04")) # EV below this gets penalty weight
STAGE4_EV_PENALTY_FACTOR = float(os.getenv("STAGE4_EV_PENALTY_FACTOR", "0.3"))        # Multiplier applied to sub-threshold scores
THESIS_MONITOR_LOOKBACK_DAYS = int(os.getenv("THESIS_MONITOR_LOOKBACK_DAYS", "2"))
THESIS_MONITOR_AUTO_RERUN = os.getenv("THESIS_MONITOR_AUTO_RERUN", "true").lower() in ("1", "true", "yes")

# ─────────────────────────────────────────────
# Multi-Agent Debate System
# ─────────────────────────────────────────────
DEBATE_ROUNDS = 3
DEBATE_AGENTS = ["growth", "value", "macro", "risk"]
DEBATE_MAX_WORKERS = int(os.getenv("DEBATE_MAX_WORKERS", "4"))     # Parallel agents per debate round
DEBATE_BETWEEN_STOCKS_SLEEP = float(os.getenv("DEBATE_BETWEEN_STOCKS_SLEEP", "0"))
# Engine: "langgraph" (deterministic sequential rounds) or "ag2" (GroupChat natural debate)
DEBATE_ENGINE = os.getenv("DEBATE_ENGINE", "langgraph").strip().lower()

# Selection Pressure (reduce false positives)
EV_POSITIVE_TARGET_PCT = float(os.getenv("EV_POSITIVE_TARGET_PCT", "0.30"))           # Only 30% should have positive EV
EV_NORMALIZATION_ENABLED = os.getenv("EV_NORMALIZATION_ENABLED", "true").lower() in ("1", "true", "yes")
EV_TOP_PCT_TO_PORTFOLIO = float(os.getenv("EV_TOP_PCT_TO_PORTFOLIO", "0.30"))         # Only top 30% EV proceed to portfolio

# Trade Logic Thresholds
EV_BUY_THRESHOLD = float(os.getenv("EV_BUY_THRESHOLD", "0.05"))                      # Min EV to trigger BUY
CONSENSUS_BUY_THRESHOLD = float(os.getenv("CONSENSUS_BUY_THRESHOLD", "0.5"))          # Min consensus to BUY
EV_SELL_THRESHOLD = float(os.getenv("EV_SELL_THRESHOLD", "0.0"))                      # Below this → SELL

# Agent Memory & Scoring
AGENT_WEIGHT_MIN = float(os.getenv("AGENT_WEIGHT_MIN", "0.5"))
AGENT_WEIGHT_MAX = float(os.getenv("AGENT_WEIGHT_MAX", "2.0"))
AGENT_MEMORY_ENABLED = os.getenv("AGENT_MEMORY_ENABLED", "true").lower() in ("1", "true", "yes")

# ─────────────────────────────────────────────
# ETL (pre-Stage research memory pipeline)
# ─────────────────────────────────────────────
ETL_LOOKBACK_DAYS = int(os.getenv("ETL_LOOKBACK_DAYS", "35"))
ETL_MAX_DOCS_PER_RUN = int(os.getenv("ETL_MAX_DOCS_PER_RUN", "150"))
ETL_MAX_JOBS_PER_RUN = int(os.getenv("ETL_MAX_JOBS_PER_RUN", "30"))
ETL_CHUNK_CHARS = int(os.getenv("ETL_CHUNK_CHARS", "2800"))
ETL_MAX_CHUNKS_PER_DOC = int(os.getenv("ETL_MAX_CHUNKS_PER_DOC", "4"))
ETL_MEMO_MODEL = os.getenv("ETL_MEMO_MODEL", "mistral:mistral-small-latest")
ETL_MEMO_MAX_TOKENS = int(os.getenv("ETL_MEMO_MAX_TOKENS", str(MAX_TOKENS)))

# Token-efficient synthesis budgets (Stage 2 -> Stage 3)
SYNTHESIS_MAX_BULL_POINTS = int(os.getenv("SYNTHESIS_MAX_BULL_POINTS", "5"))
SYNTHESIS_MAX_BEAR_POINTS = int(os.getenv("SYNTHESIS_MAX_BEAR_POINTS", "5"))
SYNTHESIS_MAX_CATALYSTS_90D = int(os.getenv("SYNTHESIS_MAX_CATALYSTS_90D", "6"))
SYNTHESIS_MAX_RISKS = int(os.getenv("SYNTHESIS_MAX_RISKS", "6"))
SYNTHESIS_MAX_UNCERTAINTIES = int(os.getenv("SYNTHESIS_MAX_UNCERTAINTIES", "4"))
SYNTHESIS_MAX_MISSING_INJECT = int(os.getenv("SYNTHESIS_MAX_MISSING_INJECT", "3"))

# ─────────────────────────────────────────────
# Data Caching TTLs (seconds)
# ─────────────────────────────────────────────
CACHE_TTL_FUNDAMENTALS = 6 * 3600        # 6 hours
CACHE_TTL_NEWS = 1 * 3600                # 1 hour
CACHE_TTL_UNIVERSE = 24 * 3600           # 24 hours
CACHE_TTL_MACRO = 4 * 3600              # 4 hours
CACHE_TTL_LIVE_PRICES = int(os.getenv("CACHE_TTL_LIVE_PRICES", "300"))  # 5 minutes

# Live price settings (yfinance history-based)
LIVE_PRICE_INTERVAL = os.getenv("LIVE_PRICE_INTERVAL", "5m")  # 1m,2m,5m,15m,30m,60m,90m,1h,1d
LIVE_PRICE_PERIOD = os.getenv("LIVE_PRICE_PERIOD", "5d")      # enough bars for interval
LIVE_PRICE_MAX_AGE_MINUTES = int(os.getenv("LIVE_PRICE_MAX_AGE_MINUTES", "60"))

# ─────────────────────────────────────────────
# News Sources (RSS feeds for last-7-day news)
# ─────────────────────────────────────────────
NEWS_RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://feeds.feedburner.com/ndtvprofit-latest",
    "https://www.livemint.com/rss/markets",
]

# Optional analyst/broker report feeds (RSS/Atom). Add your sources here.
# These are treated like "reports" (not general news) and can be used in research/monitoring.
# Example:
# ANALYST_RSS_FEEDS = ["https://example.com/broker-research/rss"]
ANALYST_RSS_FEEDS = [u.strip() for u in os.getenv("ANALYST_RSS_FEEDS", "").split(",") if u.strip()]
ANALYST_HTML_SOURCES = [u.strip() for u in os.getenv("ANALYST_HTML_SOURCES", "").split(",") if u.strip()]
# Recommended default (can be overridden in .env)
if not ANALYST_HTML_SOURCES:
    ANALYST_HTML_SOURCES = ["https://www.business-standard.com/markets/research-report"]

# Analyst report PDF ingestion (e.g., Business Standard broker tips PDFs on bsmedia.*)
ANALYST_PDF_MAX = int(os.getenv("ANALYST_PDF_MAX", "3"))
ANALYST_PDF_EXCERPT_CHARS = int(os.getenv("ANALYST_PDF_EXCERPT_CHARS", "1500"))
ANALYST_PDF_CACHE_TTL = int(os.getenv("ANALYST_PDF_CACHE_TTL", str(12 * 3600)))

# Business Standard access (some environments get HTTP 403). If you have a valid cookie/session,
# set it here to enable direct scraping of the research-report table including bsmedia PDF links.
BUSINESS_STANDARD_COOKIE = os.getenv("BUSINESS_STANDARD_COOKIE", "")

# ─────────────────────────────────────────────
# NiftyIndices Reports (PDF/HTML)
# ─────────────────────────────────────────────
NIFTYINDICES_RESEARCH_PAPERS_URL = os.getenv("NIFTYINDICES_RESEARCH_PAPERS_URL", "https://www.niftyindices.com/reports/research-paper")
NIFTYINDICES_DAILY_REPORTS_URL = os.getenv("NIFTYINDICES_DAILY_REPORTS_URL", "https://www.niftyindices.com/reports/daily-reports")
NIFTYINDICES_MONTHLY_REPORTS_URL = os.getenv("NIFTYINDICES_MONTHLY_REPORTS_URL", "https://www.niftyindices.com/reports/monthly-reports")
REPORTS_CACHE_DIR = STORAGE_DIR / "reports_cache"
REPORTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_REPORTS = int(os.getenv("CACHE_TTL_REPORTS", str(12 * 3600)))  # 12 hours
REPORTS_MAX_PER_CATEGORY = int(os.getenv("REPORTS_MAX_PER_CATEGORY", "5"))
REPORTS_TEXT_EXCERPT_CHARS = int(os.getenv("REPORTS_TEXT_EXCERPT_CHARS", "2000"))
REPORTS_ENABLED = os.getenv("REPORTS_ENABLED", "true").lower() in ("1", "true", "yes")
REPORTS_HTTP_TIMEOUT_SECONDS = int(os.getenv("REPORTS_HTTP_TIMEOUT_SECONDS", "8"))
REPORTS_HTTP_RETRIES = int(os.getenv("REPORTS_HTTP_RETRIES", "1"))

# Optional NewsAPI live feed settings
NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"
NEWSAPI_TOP_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
NEWSAPI_COUNTRY = "in"
NEWSAPI_LANGUAGE = "en"
NEWSAPI_PAGE_SIZE = 30
HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "25"))

# ─────────────────────────────────────────────
# Official Exchange / Macro Sources
# ─────────────────────────────────────────────
SOURCES_ENABLED = os.getenv("SOURCES_ENABLED", "true").lower() in ("1", "true", "yes")
CACHE_TTL_OFFICIAL_SOURCES = int(os.getenv("CACHE_TTL_OFFICIAL_SOURCES", str(60 * 60)))
SOURCE_MAX_ROWS_PER_TABLE = int(os.getenv("SOURCE_MAX_ROWS_PER_TABLE", "5000"))
OFFICIAL_SOURCE_BROWSER_FALLBACK = os.getenv("OFFICIAL_SOURCE_BROWSER_FALLBACK", "true").lower() in ("1", "true", "yes")
OFFICIAL_SOURCE_BROWSER_USER_DATA_DIR = os.getenv(
    "OFFICIAL_SOURCE_BROWSER_USER_DATA_DIR",
    str(STORAGE_DIR / "nse_profile"),
)
OFFICIAL_SOURCE_BROWSER_HEADLESS = os.getenv("OFFICIAL_SOURCE_BROWSER_HEADLESS", "true").lower() in ("1", "true", "yes")
OFFICIAL_SOURCE_BROWSER_SLOW_MO_MS = int(os.getenv("OFFICIAL_SOURCE_BROWSER_SLOW_MO_MS", "0"))
OFFICIAL_SOURCE_BROWSER_WAIT_MS = int(os.getenv("OFFICIAL_SOURCE_BROWSER_WAIT_MS", "2000"))
OFFICIAL_SOURCE_BROWSER_PAUSE_FOR_USER = os.getenv("OFFICIAL_SOURCE_BROWSER_PAUSE_FOR_USER", "false").lower() in ("1", "true", "yes")

NSE_EVENT_CALENDAR_URL = os.getenv(
    "NSE_EVENT_CALENDAR_URL",
    "https://www.nseindia.com/api/event-calendar?csv=true&index=equities",
)
NSE_FILINGS_ANNOUNCEMENTS_PAGE_URL = os.getenv(
    "NSE_FILINGS_ANNOUNCEMENTS_PAGE_URL",
    "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
)
NSE_EVENT_CALENDAR_PAGE_URL = os.getenv(
    "NSE_EVENT_CALENDAR_PAGE_URL",
    "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar",
)
NSE_CORPORATE_ACTIONS_URL = os.getenv(
    "NSE_CORPORATE_ACTIONS_URL",
    "https://www.nseindia.com/api/corporates-corporateactions?index=equities",
)
NSE_FINANCIAL_RESULTS_URL = os.getenv(
    "NSE_FINANCIAL_RESULTS_URL",
    "https://www.nseindia.com/api/corporates-financial-results?index=equities&period=Quarterly",
)
NSE_BLOCK_DEALS_URL = os.getenv(
    "NSE_BLOCK_DEALS_URL",
    "https://www.nseindia.com/api/block-deal",
)
NSE_BULK_DEALS_URL = os.getenv(
    "NSE_BULK_DEALS_URL",
    "https://www.nseindia.com/api/snapshot-capital-market-largedeal",
)
NSE_INDEX_NAMES_URL = os.getenv(
    "NSE_INDEX_NAMES_URL",
    "https://www.nseindia.com/api/index-names",
)
NSE_EQUITY_MASTER_URL = os.getenv(
    "NSE_EQUITY_MASTER_URL",
    "https://www.nseindia.com/api/equity-master",
)
NSE_UNDERLYING_INFO_URL = os.getenv(
    "NSE_UNDERLYING_INFO_URL",
    "https://www.nseindia.com/api/underlying-information",
)

# ─────────────────────────────────────────────
# Rebalancing Thresholds
# ─────────────────────────────────────────────
MIN_EV_IMPROVEMENT_TO_SWAP = 0.05        # Don't swap unless new stock has 5% higher EV
MAX_PORTFOLIO_TURNOVER_PER_CYCLE = 0.40  # Max 40% of portfolio changed per rebalance

# ─────────────────────────────────────────────
# India-specific Transaction Costs
# ─────────────────────────────────────────────
STT_RATE = 0.001          # 0.1% Securities Transaction Tax on sell
BROKERAGE_RATE = 0.0003   # 0.03% (Zerodha flat fee approximate)
IMPACT_COST = 0.001       # 0.1% estimated market impact
TOTAL_TRANSACTION_COST = STT_RATE + BROKERAGE_RATE + IMPACT_COST
