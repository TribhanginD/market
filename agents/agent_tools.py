"""
Per-agent tool registry. Each domain agent (growth, value, macro, risk) gets its own
specialized toolset and dispatcher for the Anthropic agentic loop.

Tools wrap data/ fetchers and persistence/ SQLite reads. Agents pull live data
during debate instead of relying on pre-fetched context.
"""

import json
import logging


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Tool schemas (Anthropic format)
# ──────────────────────────────────────────────────────────

TOOL_GET_FUNDAMENTALS = {
    "name": "get_fundamentals",
    "description": "Fetch full fundamentals for a stock: PE, PB, ROE, revenue/earnings growth, debt, "
                   "margins, technicals (RSI, SMA, drawdown), 52W return, market cap, analyst targets.",
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "NSE symbol, no .NS suffix"},
        },
        "required": ["symbol"],
    },
}

TOOL_GET_NEWS = {
    "name": "get_recent_news",
    "description": "Fetch recent news for a stock (up to 14 days). Returns titles, summaries, sources, dates.",
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "company_name": {"type": "string"},
            "days": {"type": "integer", "description": "Lookback window, default 7, max 14"},
        },
        "required": ["symbol", "company_name"],
    },
}

TOOL_GET_MACRO = {
    "name": "get_macro_context",
    "description": "Current Indian macro: Nifty 50 level + 1m/3m return, USD/INR, India VIX, crude oil, "
                   "RBI repo rate context, sector index returns.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

TOOL_GET_FII_FLOWS = {
    "name": "get_fii_dii_flows",
    "description": "Recent FII (foreign institutional) and DII (domestic institutional) net flows in INR cr. "
                   "Indicates institutional sentiment direction.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

TOOL_GET_SECTOR_PEERS = {
    "name": "get_sector_peer_valuations",
    "description": "Compare a stock's PE/PB/ROE vs sector peers from latest universe data. "
                   "Returns peer PE distribution (median, 25th/75th percentile) for the same sector.",
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "sector": {"type": "string"},
        },
        "required": ["symbol", "sector"],
    },
}

TOOL_GET_FILINGS = {
    "name": "query_filings",
    "description": "Query recent NSE/BSE filings for a stock. Returns filing types (results, governance, "
                   "audit, related-party, regulatory). Lookback default 90 days.",
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "days": {"type": "integer", "description": "Lookback days, default 90"},
        },
        "required": ["symbol"],
    },
}

TOOL_GET_CORP_ACTIONS = {
    "name": "query_corporate_actions",
    "description": "Query corporate actions (dividends, splits, buybacks, bonus, rights). "
                   "Useful for risk: heavy promoter selling, equity dilution, capital structure changes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "days": {"type": "integer", "description": "Lookback days, default 180"},
        },
        "required": ["symbol"],
    },
}

TOOL_GET_BULK_DEALS = {
    "name": "query_bulk_block_deals",
    "description": "Query recent bulk and block deals for a stock. Indicates large institutional or "
                   "promoter buy/sell activity.",
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "days": {"type": "integer", "description": "Lookback days, default 60"},
        },
        "required": ["symbol"],
    },
}

TOOL_GET_RESULTS_CALENDAR = {
    "name": "get_upcoming_results",
    "description": "Get upcoming earnings result dates for a stock (next 60 days).",
    "input_schema": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
}

TOOL_GET_ANALYST_REPORTS = {
    "name": "get_analyst_reports",
    "description": "Recent analyst/broker reports for a stock. Returns titles, sources, dates, and excerpts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "company_name": {"type": "string"},
            "days": {"type": "integer", "description": "Lookback days, default 14"},
        },
        "required": ["symbol", "company_name"],
    },
}

TOOL_WEB_SEARCH = {
    "name": "web_search",
    "description": "Live web search via Tavily. Use for breaking news, regulatory actions, management commentary, "
                   "court rulings, competitor moves, sector trends not in cached sources. "
                   "Returns up to 5 results with title, url, snippet, plus an LLM-generated answer summary.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query. Be specific. Include company name, NSE symbol, time qualifiers if relevant."},
            "search_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "basic=fast, advanced=deeper. Default basic."},
            "max_results": {"type": "integer", "description": "Number of results, 1-10. Default 5."},
            "days": {"type": "integer", "description": "Restrict to recent N days (1-30). Optional."},
        },
        "required": ["query"],
    },
}


TOOL_GET_SYMBOL_MEMORY = {
    "name": "get_symbol_memory",
    "description": "Retrieve compact research memory for a stock: prior memos, event counts, recent ETL summary. "
                   "Useful to check if prior debates flagged risks or theses.",
    "input_schema": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
}


# ──────────────────────────────────────────────────────────
# Per-agent toolsets
# ──────────────────────────────────────────────────────────

GROWTH_TOOLS = [
    TOOL_GET_FUNDAMENTALS,
    TOOL_GET_NEWS,
    TOOL_GET_RESULTS_CALENDAR,
    TOOL_GET_ANALYST_REPORTS,
    TOOL_GET_SYMBOL_MEMORY,
    TOOL_WEB_SEARCH,
]

VALUE_TOOLS = [
    TOOL_GET_FUNDAMENTALS,
    TOOL_GET_SECTOR_PEERS,
    TOOL_GET_NEWS,
    TOOL_GET_ANALYST_REPORTS,
    TOOL_GET_SYMBOL_MEMORY,
    TOOL_WEB_SEARCH,
]

MACRO_TOOLS = [
    TOOL_GET_MACRO,
    TOOL_GET_FII_FLOWS,
    TOOL_GET_FUNDAMENTALS,
    TOOL_GET_NEWS,
    TOOL_GET_SECTOR_PEERS,
    TOOL_WEB_SEARCH,
]

RISK_TOOLS = [
    TOOL_GET_FUNDAMENTALS,
    TOOL_GET_FILINGS,
    TOOL_GET_CORP_ACTIONS,
    TOOL_GET_BULK_DEALS,
    TOOL_GET_NEWS,
    TOOL_GET_SYMBOL_MEMORY,
    TOOL_WEB_SEARCH,
]


# ──────────────────────────────────────────────────────────
# Dispatch (single shared dispatcher)
# ──────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict):
    """Execute a tool call. Returns dict (JSON-serializable)."""
    try:
        if tool_name == "get_fundamentals":
            from data.fetcher import get_fundamentals
            symbol = (tool_input.get("symbol") or "").strip().upper()
            if not symbol:
                return {"error": "symbol required"}
            return _compact_fundamentals(get_fundamentals(f"{symbol}.NS"))

        if tool_name == "get_recent_news":
            from data.fetcher import get_news_for_stock
            symbol = (tool_input.get("symbol") or "").strip().upper()
            company = tool_input.get("company_name") or symbol
            days = max(1, min(int(tool_input.get("days", 7) or 7), 14))
            news = get_news_for_stock(symbol, company, days=days)
            return _compact_news(news)

        if tool_name == "get_macro_context":
            from data.fetcher import get_macro_context
            return get_macro_context()

        if tool_name == "get_fii_dii_flows":
            from data.fetcher import _fetch_fii_dii_flows
            return _fetch_fii_dii_flows()

        if tool_name == "get_sector_peer_valuations":
            return _query_sector_peers(
                (tool_input.get("symbol") or "").strip().upper(),
                tool_input.get("sector") or "",
            )

        if tool_name == "query_filings":
            return _query_table(
                "filings",
                (tool_input.get("symbol") or "").strip().upper(),
                int(tool_input.get("days", 90) or 90),
            )

        if tool_name == "query_corporate_actions":
            return _query_table(
                "corporate_actions",
                (tool_input.get("symbol") or "").strip().upper(),
                int(tool_input.get("days", 180) or 180),
            )

        if tool_name == "query_bulk_block_deals":
            symbol = (tool_input.get("symbol") or "").strip().upper()
            days = int(tool_input.get("days", 60) or 60)
            bulk = _query_table("bulk_deals", symbol, days)
            block = _query_table("block_deals", symbol, days)
            return {"bulk_deals": bulk, "block_deals": block}

        if tool_name == "get_upcoming_results":
            return _query_results_calendar((tool_input.get("symbol") or "").strip().upper())

        if tool_name == "get_analyst_reports":
            from data.analyst_reports import get_analyst_reports_for_stock
            symbol = (tool_input.get("symbol") or "").strip().upper()
            company = tool_input.get("company_name") or symbol
            days = max(1, min(int(tool_input.get("days", 14) or 14), 30))
            reports = get_analyst_reports_for_stock(symbol, company, days=days)[:6]
            return reports

        if tool_name == "get_symbol_memory":
            return _query_symbol_memory((tool_input.get("symbol") or "").strip().upper())

        if tool_name == "web_search":
            return _tavily_search(
                query=tool_input.get("query") or "",
                search_depth=tool_input.get("search_depth") or "basic",
                max_results=int(tool_input.get("max_results") or 5),
                days=tool_input.get("days"),
            )

        return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        logger.warning(f"Tool {tool_name} failed: {e}")
        return {"error": str(e), "tool": tool_name}


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

_FUNDAMENTAL_KEEP_FIELDS = [
    "symbol", "shortName", "longName", "sector", "industry",
    "price", "currentPrice", "previousClose", "marketCap",
    "trailingPE", "forwardPE", "priceToBook", "returnOnEquity",
    "revenueGrowth", "earningsGrowth", "debtToEquity",
    "operatingMargins", "profitMargins", "ebitdaMargins",
    "currentRatio", "quickRatio",
    "targetMeanPrice", "recommendationKey", "numberOfAnalystOpinions",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "52w_return_pct", "6m_return_pct",
    "rsi_14", "sma_20", "sma_50", "sma_200",
    "drawdown_from_6m_high", "volatility_30d_annualized", "price_vs_52w_high",
    "annual_total_revenue", "annual_net_income",
    "quarter_total_revenue", "quarter_net_income",
]


def _compact_fundamentals(f: dict) -> dict:
    if not isinstance(f, dict) or "error" in f:
        return f or {}
    out = {k: f.get(k) for k in _FUNDAMENTAL_KEEP_FIELDS if k in f}
    if "price" not in out and out.get("currentPrice"):
        out["price"] = out["currentPrice"]
    return out


def _compact_news(news: list, max_items: int = 8) -> list:
    if not isinstance(news, list):
        return []
    out = []
    for a in news[:max_items]:
        if not isinstance(a, dict):
            continue
        out.append({
            "title": a.get("title"),
            "source": a.get("source"),
            "published_at": a.get("published_at") or a.get("date"),
            "summary": (a.get("summary") or "")[:240],
        })
    return out


def _query_table(table: str, symbol: str, days: int) -> list:
    """Generic SQLite query for normalized source tables filtered by symbol + recent days."""
    if not symbol:
        return []
    try:
        from persistence.db import fetch_rows
        # Tables vary in column names; try common ones
        query = f"""
            SELECT * FROM {table}
            WHERE (symbol = ? OR symbol = ? OR isin LIKE ?)
              AND (event_date >= date('now', ?) OR filing_date >= date('now', ?) OR record_date >= date('now', ?))
            ORDER BY COALESCE(event_date, filing_date, record_date) DESC
            LIMIT 25
        """
        days_clause = f"-{int(days)} days"
        try:
            rows = fetch_rows(query, (symbol, symbol.upper(), f"%{symbol}%", days_clause, days_clause, days_clause))
        except Exception:
            # Fallback: simpler query without date filter
            rows = fetch_rows(
                f"SELECT * FROM {table} WHERE symbol = ? ORDER BY id DESC LIMIT 15",
                (symbol,),
            )
        return [_strip_payload(r) for r in rows[:15]]
    except Exception as e:
        return [{"error": str(e), "table": table}]


def _query_results_calendar(symbol: str) -> list:
    if not symbol:
        return []
    try:
        from persistence.db import fetch_rows
        rows = fetch_rows(
            "SELECT * FROM results_calendar WHERE symbol = ? AND result_date >= date('now') "
            "ORDER BY result_date ASC LIMIT 5",
            (symbol,),
        )
        return [_strip_payload(r) for r in rows]
    except Exception as e:
        return [{"error": str(e)}]


def _query_symbol_memory(symbol: str) -> dict:
    if not symbol:
        return {}
    try:
        from persistence.db import fetch_rows
        rows = fetch_rows(
            "SELECT symbol, company, summary_json, updated_at FROM symbol_memory WHERE symbol = ? LIMIT 1",
            (symbol,),
        )
        if not rows:
            return {"symbol": symbol, "memory": None}
        row = rows[0]
        try:
            summary = json.loads(row.get("summary_json") or "{}")
        except Exception:
            summary = {}
        return {
            "symbol": row.get("symbol"),
            "company": row.get("company"),
            "updated_at": row.get("updated_at"),
            "memory": summary,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def _query_sector_peers(symbol: str, sector: str) -> dict:
    """Compute peer PE/PB/ROE distribution for a sector from cached fundamentals."""
    if not sector:
        return {"error": "sector required"}
    try:
        # Pull from universe + fundamentals cache
        from data.fetcher import get_fundamentals
        from data.nifty500 import get_nifty500
        import statistics

        df = get_nifty500()
        if df is None or df.empty:
            return {"error": "universe unavailable"}
        sector_lower = sector.lower()
        peer_df = df[df["sector"].fillna("").str.lower() == sector_lower]
        peer_df = peer_df[peer_df["symbol"] != symbol].head(25)

        pe_vals, pb_vals, roe_vals = [], [], []
        for _, row in peer_df.iterrows():
            yf_sym = row.get("yf_symbol") or f"{row.get('symbol')}.NS"
            f = get_fundamentals(yf_sym)
            if not isinstance(f, dict):
                continue
            pe = f.get("trailingPE")
            pb = f.get("priceToBook")
            roe = f.get("returnOnEquity")
            if isinstance(pe, (int, float)) and pe > 0:
                pe_vals.append(pe)
            if isinstance(pb, (int, float)) and pb > 0:
                pb_vals.append(pb)
            if isinstance(roe, (int, float)):
                roe_vals.append(roe)

        def _stats(vals):
            if not vals:
                return None
            vals_sorted = sorted(vals)
            return {
                "median": round(statistics.median(vals_sorted), 3),
                "p25": round(vals_sorted[len(vals_sorted) // 4], 3),
                "p75": round(vals_sorted[(3 * len(vals_sorted)) // 4], 3),
                "n": len(vals_sorted),
            }

        return {
            "symbol": symbol,
            "sector": sector,
            "peer_count": int(len(peer_df)),
            "pe": _stats(pe_vals),
            "pb": _stats(pb_vals),
            "roe": _stats(roe_vals),
        }
    except Exception as e:
        return {"error": str(e)}


def _tavily_search(query: str, search_depth: str = "basic", max_results: int = 5, days=None) -> dict:
    """Tavily REST search. Requires TAVILY_API_KEY in env."""
    import config
    api_key = getattr(config, "TAVILY_API_KEY", "")
    if not api_key:
        return {"error": "TAVILY_API_KEY not set"}
    if not query or not query.strip():
        return {"error": "query required"}
    try:
        import requests
        depth = search_depth if search_depth in ("basic", "advanced", "fast", "ultra-fast") else "basic"
        payload = {
            "query": query.strip()[:400],
            "search_depth": depth,
            "max_results": max(1, min(int(max_results), 10)),
            "include_answer": True,
            "include_raw_content": False,
        }
        if days:
            try:
                payload["days"] = max(1, min(int(days), 30))
                payload["topic"] = "news"
            except Exception:
                pass
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "market-agent-tools/1.0",
        }
        resp = requests.post("https://api.tavily.com/search", json=payload, headers=headers, timeout=20)
        if resp.status_code != 200:
            return {"error": f"tavily {resp.status_code}", "body": resp.text[:300]}
        data = resp.json() or {}
        results = []
        for r in (data.get("results") or [])[:payload["max_results"]]:
            results.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": (r.get("content") or "")[:400],
                "score": r.get("score"),
                "published_date": r.get("published_date"),
            })
        return {
            "query": query,
            "answer": (data.get("answer") or "")[:800],
            "results": results,
        }
    except Exception as e:
        return {"error": str(e), "tool": "web_search"}


def _strip_payload(row: dict) -> dict:
    """Remove large blob fields (raw payloads) from row before returning to LLM."""
    if not isinstance(row, dict):
        return row
    out = {}
    for k, v in row.items():
        if k in ("payload", "raw", "raw_payload", "html", "xml_blob"):
            continue
        if isinstance(v, str) and len(v) > 400:
            out[k] = v[:400] + "..."
        else:
            out[k] = v
    return out
