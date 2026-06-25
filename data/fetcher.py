"""
Data fetcher: fundamentals, price data, news, and macro context.
All fetches go through the local cache to avoid redundant API calls.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import requests
import yfinance as yf
try:
    import feedparser
except ImportError:
    feedparser = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

import config
import data.cache as cache

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = getattr(config, "HTTP_TIMEOUT_SECONDS", 12)
NSE_BOOTSTRAP_URL = "https://www.nseindia.com"
NSE_FII_DII_ENDPOINT = "https://www.nseindia.com/api/fiidiiTradeReact"

_NSE_SESSION: Optional[requests.Session] = None
_WARNED_MISSING_FEEDPARSER = False


# ─────────────────────────────────────────────
# Fundamentals
# ─────────────────────────────────────────────

FUNDAMENTAL_FIELDS = [
    "shortName", "longName", "sector", "industry",
    "currentPrice", "previousClose", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "marketCap", "enterpriseValue",
    "trailingPE", "forwardPE", "trailingEps", "forwardEps",
    "priceToBook", "priceToSalesTrailing12Months",
    "returnOnEquity", "returnOnAssets",
    "revenueGrowth", "earningsGrowth", "revenuePerShare",
    "debtToEquity", "currentRatio", "quickRatio",
    "grossMargins", "operatingMargins", "profitMargins",
    "totalRevenue", "netIncomeToCommon", "freeCashflow",
    "dividendYield", "fiveYearAvgDividendYield",
    "52WeekChange", "beta",
    "recommendationKey", "numberOfAnalystOpinions",
    "targetHighPrice", "targetLowPrice", "targetMeanPrice", "targetMedianPrice",
    "sharesOutstanding", "floatShares", "heldPercentInstitutions",
]


def get_fundamentals(yf_symbol: str, force_refresh: bool = False) -> dict:
    """
    Fetch fundamental data for a single stock from yfinance.
    Cached for CACHE_TTL_FUNDAMENTALS seconds.
    """
    cached = cache.get("fundamentals", yf_symbol, config.CACHE_TTL_FUNDAMENTALS)
    if cached is not None and not force_refresh:
        return cached

    try:
        ticker = yf.Ticker(yf_symbol)
        info = {}
        try:
            info = ticker.fast_info or {}
        except Exception:
            info = {}

        # `fast_info` is lightweight and good for price/volume/market cap-ish fields.
        # For ratios/growth metrics we still need `info`, but we only pull it if needed.
        try:
            raw_info = ticker.info or {}
        except Exception:
            raw_info = {}

        merged = dict(raw_info)
        merged.update({k: v for k, v in info.items() if v is not None})
        data = {field: merged.get(field) for field in FUNDAMENTAL_FIELDS}
        data["yf_symbol"] = yf_symbol
        data["fetch_time"] = datetime.now().isoformat()
        data["52w_return_pct"] = data.get("52WeekChange")
        data["price"] = None

        # Prefer history-derived "live" price to avoid stale `info.currentPrice`.
        live = get_live_price(yf_symbol)
        if live.get("price") is not None:
            data["price"] = live.get("price")
            data["price_timestamp"] = live.get("timestamp")
        else:
            data["price"] = data.get("currentPrice") or data.get("previousClose")

        try:
            hist = ticker.history(period="6mo")
            if not hist.empty:
                close_start = _safe_float(hist["Close"].iloc[0])
                close_end = _safe_float(hist["Close"].iloc[-1])
                if close_start and close_end:
                    data["6m_return_pct"] = (close_end - close_start) / close_start
                else:
                    data["6m_return_pct"] = None

                data["avg_volume_30d"] = _safe_float(hist["Volume"].tail(30).mean())
                if not data.get("price"):
                    data["price"] = close_end

                fifty_two_week_high = _safe_float(data.get("fiftyTwoWeekHigh"))
                if data.get("price") and fifty_two_week_high:
                    data["price_vs_52w_high"] = data["price"] / fifty_two_week_high
                else:
                    data["price_vs_52w_high"] = None

                # ── Technicals (non-LLM stock report enrichment) ──
                try:
                    import numpy as np

                    closes = hist["Close"].astype(float)
                    rets = closes.pct_change()
                    data["volatility_30d_annualized"] = float(rets.tail(30).std() * (252 ** 0.5)) if len(rets.dropna()) >= 10 else None

                    # Simple moving averages
                    data["sma_20"] = _safe_float(closes.rolling(20).mean().iloc[-1]) if len(closes) >= 20 else None
                    data["sma_50"] = _safe_float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None
                    data["sma_200"] = _safe_float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None

                    # RSI(14)
                    if len(closes) >= 15:
                        delta = closes.diff()
                        up = delta.clip(lower=0).rolling(14).mean()
                        down = (-delta.clip(upper=0)).rolling(14).mean()
                        rs = up / down.replace(0, np.nan)
                        rsi = 100 - (100 / (1 + rs))
                        data["rsi_14"] = _safe_float(rsi.iloc[-1])
                    else:
                        data["rsi_14"] = None

                    # Drawdown from max close in window
                    roll_max = closes.cummax()
                    dd = (closes / roll_max) - 1.0
                    data["drawdown_from_6m_high"] = _safe_float(dd.iloc[-1])
                except Exception:
                    data["volatility_30d_annualized"] = None
                    data["sma_20"] = None
                    data["sma_50"] = None
                    data["sma_200"] = None
                    data["rsi_14"] = None
                    data["drawdown_from_6m_high"] = None
            else:
                data["6m_return_pct"] = None
                data["avg_volume_30d"] = None
                data["price_vs_52w_high"] = None
        except Exception:
            data["6m_return_pct"] = None
            data["avg_volume_30d"] = None
            data["price_vs_52w_high"] = None

        # ── Financial statement summary (best-effort, small) ──
        try:
            fin = getattr(ticker, "financials", None)
            qfin = getattr(ticker, "quarterly_financials", None)

            def _extract_latest(df, key: str) -> Optional[float]:
                try:
                    if df is None or getattr(df, "empty", True):
                        return None
                    row = df.loc[key] if key in df.index else None
                    if row is None:
                        return None
                    # Most recent is first column in yfinance financials
                    return _safe_float(row.iloc[0])
                except Exception:
                    return None

            # Common labels vary; try a few.
            data["annual_total_revenue"] = _extract_latest(fin, "Total Revenue") or _extract_latest(fin, "TotalRevenue")
            data["annual_net_income"] = _extract_latest(fin, "Net Income") or _extract_latest(fin, "NetIncome")
            data["quarter_total_revenue"] = _extract_latest(qfin, "Total Revenue") or _extract_latest(qfin, "TotalRevenue")
            data["quarter_net_income"] = _extract_latest(qfin, "Net Income") or _extract_latest(qfin, "NetIncome")
        except Exception:
            data["annual_total_revenue"] = None
            data["annual_net_income"] = None
            data["quarter_total_revenue"] = None
            data["quarter_net_income"] = None

        # ── Calendar / earnings ──
        try:
            cal = getattr(ticker, "calendar", None)
            # calendar is typically a DataFrame; keep a tiny JSON-friendly version
            if cal is not None and hasattr(cal, "to_dict"):
                data["calendar"] = cal.to_dict()
            else:
                data["calendar"] = None
        except Exception:
            data["calendar"] = None

        cache.set("fundamentals", yf_symbol, data)
        logger.debug(f"Fetched fundamentals for {yf_symbol}")
        return data

    except Exception as e:
        logger.warning(f"Failed to fetch fundamentals for {yf_symbol}: {e}")
        return {"yf_symbol": yf_symbol, "error": str(e)}


def get_fundamentals_batch(yf_symbols: list[str], delay_seconds: float = 0.3) -> dict[str, dict]:
    """
    Fetch fundamentals for multiple stocks with rate-limiting.
    Returns dict keyed by yf_symbol.
    """
    results: dict[str, dict] = {}
    total = len(yf_symbols)
    if total == 0:
        return results

    # Batch history download for prices/returns/volume.
    batch_hist = _download_history_batch(yf_symbols, period="6mo", interval="1d")
    live_prices = get_live_prices_batch(yf_symbols)

    for i, sym in enumerate(yf_symbols):
        if i > 0 and i % 50 == 0:
            logger.info(f"  Fetched {i}/{total} fundamentals...")

        results[sym] = get_fundamentals(sym)

        # Override price with live history-derived data when available.
        live = live_prices.get(sym) or {}
        if live.get("price") is not None:
            results[sym]["price"] = live.get("price")
            results[sym]["price_timestamp"] = live.get("timestamp")

        # Attach return/volume derived from batch history to reduce per-ticker variability.
        hist = batch_hist.get(sym)
        if hist is not None and not hist.empty:
            close_start = _safe_float(hist["Close"].iloc[0])
            close_end = _safe_float(hist["Close"].iloc[-1])
            if close_start and close_end:
                results[sym]["6m_return_pct"] = (close_end - close_start) / close_start
            results[sym]["avg_volume_30d"] = _safe_float(hist["Volume"].tail(30).mean())

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    logger.info(
        "Fetched fundamentals for %s stocks (%s successful)",
        len(results),
        sum(1 for v in results.values() if "error" not in v),
    )
    return results


# ─────────────────────────────────────────────
# Live Prices (yfinance)
# ─────────────────────────────────────────────

def get_live_price(yf_symbol: str, force_refresh: bool = False) -> dict:
    """
    Live-ish price via yfinance history.
    Returns: { "yf_symbol": ..., "price": float|None, "timestamp": iso|"" }
    """
    cached = cache.get("live_price", yf_symbol, config.CACHE_TTL_LIVE_PRICES)
    if cached is not None and not force_refresh:
        return cached

    out = {"yf_symbol": yf_symbol, "price": None, "timestamp": ""}
    try:
        hist = yf.download(
            yf_symbol,
            period=config.LIVE_PRICE_PERIOD,
            interval=config.LIVE_PRICE_INTERVAL,
            progress=False,
            threads=False,
        )
        if hist is None or hist.empty:
            cache.set("live_price", yf_symbol, out)
            return out

        # yfinance returns tz-aware index sometimes; normalize to naive ISO.
        ts = hist.index[-1]
        try:
            ts_dt = ts.to_pydatetime()
        except Exception:
            ts_dt = None
        if ts_dt is not None and ts_dt.tzinfo is not None:
            ts_dt = ts_dt.astimezone(timezone.utc).replace(tzinfo=None)

        price = _safe_float(hist["Close"].iloc[-1])
        out["price"] = price
        out["timestamp"] = ts_dt.isoformat() if ts_dt else str(ts)
    except Exception as e:
        out["error"] = str(e)

    cache.set("live_price", yf_symbol, out)
    return out


def get_live_prices_batch(yf_symbols: list[str]) -> dict[str, dict]:
    """
    Batch live prices via a single yfinance.download call.
    Returns dict keyed by yf_symbol.
    """
    key = "batch_" + str(hash(tuple(sorted(yf_symbols))))
    cached = cache.get("live_prices_batch", key, config.CACHE_TTL_LIVE_PRICES)
    if cached is not None:
        return cached

    results: dict[str, dict] = {sym: {"yf_symbol": sym, "price": None, "timestamp": ""} for sym in yf_symbols}
    if not yf_symbols:
        return results

    try:
        hist = yf.download(
            yf_symbols,
            period=config.LIVE_PRICE_PERIOD,
            interval=config.LIVE_PRICE_INTERVAL,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if hist is None or hist.empty:
            cache.set("live_prices_batch", key, results)
            return results

        # Multi-ticker downloads may return either:
        # - column MultiIndex: (PriceField, Ticker) or (Ticker, PriceField), depending on version
        # - single-index for single symbol
        ts = hist.index[-1]
        try:
            ts_dt = ts.to_pydatetime()
        except Exception:
            ts_dt = None
        if ts_dt is not None and ts_dt.tzinfo is not None:
            ts_dt = ts_dt.astimezone(timezone.utc).replace(tzinfo=None)
        ts_str = ts_dt.isoformat() if ts_dt else str(ts)

        if hasattr(hist.columns, "nlevels") and hist.columns.nlevels == 2:
            # Try (Ticker, Field) first, then (Field, Ticker).
            level0 = list(hist.columns.levels[0])
            if any(str(x) in set(yf_symbols) for x in level0):
                for sym in yf_symbols:
                    try:
                        price = _safe_float(hist[(sym, "Close")].iloc[-1])
                        results[sym] = {"yf_symbol": sym, "price": price, "timestamp": ts_str}
                    except Exception:
                        continue
            else:
                for sym in yf_symbols:
                    try:
                        price = _safe_float(hist[("Close", sym)].iloc[-1])
                        results[sym] = {"yf_symbol": sym, "price": price, "timestamp": ts_str}
                    except Exception:
                        continue
        else:
            # Single symbol dataframe
            if len(yf_symbols) == 1:
                sym = yf_symbols[0]
                price = _safe_float(hist["Close"].iloc[-1])
                results[sym] = {"yf_symbol": sym, "price": price, "timestamp": ts_str}
    except Exception as e:
        for sym in yf_symbols:
            results[sym]["error"] = str(e)

    cache.set("live_prices_batch", key, results)
    return results


def _download_history_batch(yf_symbols: list[str], period: str, interval: str):
    """
    Batch download historical OHLCV and return per-symbol DataFrames.
    """
    results: dict[str, Any] = {}
    if not yf_symbols:
        return results

    try:
        hist = yf.download(
            yf_symbols,
            period=period,
            interval=interval,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if hist is None or hist.empty:
            return results
        if hasattr(hist.columns, "nlevels") and hist.columns.nlevels == 2:
            level0 = list(hist.columns.levels[0])
            if any(str(x) in set(yf_symbols) for x in level0):
                for sym in yf_symbols:
                    try:
                        df = hist[sym].dropna(how="all")
                        results[sym] = df
                    except Exception:
                        continue
            else:
                # (Field, Ticker)
                for sym in yf_symbols:
                    try:
                        df = hist.xs(sym, level=1, axis=1).dropna(how="all")
                        results[sym] = df
                    except Exception:
                        continue
        else:
            if len(yf_symbols) == 1:
                results[yf_symbols[0]] = hist.dropna(how="all")
        return results
    except Exception:
        return results


# ─────────────────────────────────────────────
# News Fetching
# ─────────────────────────────────────────────

def get_news_for_stock(
    symbol: str,
    company_name: str,
    days: int = 7,
    max_articles: int = 20,
) -> list[dict]:
    """
    Fetch recent news for a stock from multiple live sources:
    - RSS feeds
    - Yahoo Finance ticker news
    - NewsAPI (if NEWS_API_KEY is configured)
    """
    if getattr(config, "CAVEMAN_MODE", False):
        max_articles = min(max_articles, 5)

    cache_key = f"{symbol}_{days}d"
    cached = cache.get("news", cache_key, config.CACHE_TTL_NEWS)
    if cached is not None:
        return cached

    cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    search_terms = _build_search_terms(symbol, company_name)

    articles = []
    articles.extend(_fetch_stock_news_from_google(symbol, company_name, cutoff_date, max_articles * 3))
    articles.extend(_fetch_stock_news_from_rss(search_terms, cutoff_date, max_articles * 3))
    articles.extend(_fetch_stock_news_from_yahoo(symbol, search_terms, cutoff_date, max_articles * 2))
    articles.extend(_fetch_stock_news_from_newsapi(symbol, company_name, search_terms, cutoff_date, max_articles * 2))

    unique_articles = _dedupe_sort_limit_articles(
        articles=articles,
        cutoff_date=cutoff_date,
        max_articles=max_articles,
    )

    cache.set("news", cache_key, unique_articles)
    return unique_articles


def _fetch_stock_news_from_google(
    symbol: str,
    company_name: str,
    cutoff_date: datetime,
    max_articles: int,
) -> list[dict]:
    """
    Google News RSS search (no API key) tends to be the most reliable way to get
    stock-specific headlines when publisher RSS feeds are broad and don't match
    per-ticker keywords.
    """
    queries = [
        f"({company_name} OR {symbol}) stock NSE",
        f"{company_name} earnings",
        f"{symbol} NSE news",
    ]
    articles: list[dict] = []
    for q in queries:
        feed_url = _google_news_rss_url(q)
        try:
            feed_source, entries = _load_rss_entries(feed_url)
            for entry in entries:
                pub_date = (
                    _parse_date(entry.get("published"))
                    or _parse_date(entry.get("updated"))
                    or _parse_date(entry.get("pubDate"))
                    or _parse_date(entry.get("published_parsed"))
                    or _parse_date(entry.get("updated_parsed"))
                )
                if pub_date and pub_date < cutoff_date:
                    continue

                article = _build_article(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", entry.get("description", "")),
                    published=pub_date,
                    source=feed_source or "Google News",
                    url=entry.get("link", ""),
                )
                if article:
                    articles.append(article)
                    if len(articles) >= max_articles:
                        return articles
        except Exception as e:
            logger.debug("Google News RSS fetch failed for %s: %s", symbol, e)
    return articles


def get_market_news(days: int = 7) -> list[dict]:
    """
    Fetch general Indian market news (not stock-specific) from:
    - RSS feeds
    - NewsAPI top headlines (optional)
    """
    cache_key = f"general_{days}d"
    cached = cache.get("market_news", cache_key, config.CACHE_TTL_NEWS)
    if cached is not None:
        return cached

    cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

    articles = []
    articles.extend(_fetch_market_news_from_rss(cutoff_date, max_articles=60))
    articles.extend(_fetch_market_news_from_newsapi(cutoff_date, max_articles=30))

    unique_articles = _dedupe_sort_limit_articles(
        articles=articles,
        cutoff_date=cutoff_date,
        max_articles=50,
    )

    cache.set("market_news", cache_key, unique_articles)
    return unique_articles


def _fetch_stock_news_from_rss(
    search_terms: list[str],
    cutoff_date: datetime,
    max_articles: int,
) -> list[dict]:
    articles = []
    for feed_url in config.NEWS_RSS_FEEDS:
        try:
            feed_source, entries = _load_rss_entries(feed_url)
            for entry in entries:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                combined_text = f"{title} {summary}".lower()
                if not _is_article_relevant(combined_text, search_terms):
                    continue

                pub_date = (
                    _parse_date(entry.get("published"))
                    or _parse_date(entry.get("updated"))
                    or _parse_date(entry.get("pubDate"))
                    or _parse_date(entry.get("published_parsed"))
                    or _parse_date(entry.get("updated_parsed"))
                )
                if pub_date and pub_date < cutoff_date:
                    continue

                article = _build_article(
                    title=title,
                    summary=summary,
                    published=pub_date,
                    source=feed_source,
                    url=entry.get("link", ""),
                )
                if article:
                    articles.append(article)
                    if len(articles) >= max_articles:
                        return articles
        except Exception as e:
            logger.warning(f"Failed to parse stock RSS feed {feed_url}: {e}")

    return articles


def _fetch_stock_news_from_yahoo(
    symbol: str,
    search_terms: list[str],
    cutoff_date: datetime,
    max_articles: int,
) -> list[dict]:
    articles = []
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        yf_news = ticker.news or []
        for item in yf_news[:max_articles * 2]:
            title = item.get("title", "")
            summary = item.get("summary", "") or item.get("content", "")

            pub_date = (
                _parse_date(item.get("providerPublishTime"))
                or _parse_date(item.get("pubDate"))
                or _parse_date(item.get("publishedAt"))
            )
            if pub_date and pub_date < cutoff_date:
                continue

            article = _build_article(
                title=title,
                summary=summary,
                published=pub_date,
                source=item.get("publisher", "Yahoo Finance"),
                url=item.get("link", ""),
            )
            if article:
                articles.append(article)
                if len(articles) >= max_articles:
                    break
    except Exception as e:
        logger.debug(f"Yahoo news fetch failed for {symbol}: {e}")

    return articles


def _fetch_stock_news_from_newsapi(
    symbol: str,
    company_name: str,
    search_terms: list[str],
    cutoff_date: datetime,
    max_articles: int,
) -> list[dict]:
    if not config.NEWS_API_KEY:
        return []

    query = _build_newsapi_stock_query(symbol, company_name)
    params = {
        "q": query,
        "language": config.NEWSAPI_LANGUAGE,
        "sortBy": "publishedAt",
        "from": cutoff_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pageSize": min(max_articles, config.NEWSAPI_PAGE_SIZE),
        "apiKey": config.NEWS_API_KEY,
    }

    try:
        response = requests.get(
            config.NEWSAPI_EVERYTHING_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            logger.debug(
                "NewsAPI stock query failed for %s: status=%s body=%s",
                symbol,
                response.status_code,
                response.text[:300],
            )
            return []

        payload = response.json()
        feed_articles = payload.get("articles", []) if isinstance(payload, dict) else []
        articles = []
        for item in feed_articles:
            title = item.get("title", "")
            summary = item.get("description", "") or item.get("content", "")
            combined_text = f"{title} {summary}".lower()
            if not _is_article_relevant(combined_text, search_terms):
                continue

            pub_date = _parse_date(item.get("publishedAt"))
            if pub_date and pub_date < cutoff_date:
                continue

            source_name = ""
            source_obj = item.get("source")
            if isinstance(source_obj, dict):
                source_name = source_obj.get("name", "")

            article = _build_article(
                title=title,
                summary=summary,
                published=pub_date,
                source=source_name or "NewsAPI",
                url=item.get("url", ""),
            )
            if article:
                articles.append(article)

        return articles[:max_articles]
    except Exception as e:
        logger.debug(f"NewsAPI stock fetch failed for {symbol}: {e}")
        return []


def _fetch_market_news_from_rss(cutoff_date: datetime, max_articles: int) -> list[dict]:
    articles = []
    for feed_url in config.NEWS_RSS_FEEDS:
        try:
            feed_source, entries = _load_rss_entries(feed_url)
            for entry in entries[:50]:
                pub_date = (
                    _parse_date(entry.get("published"))
                    or _parse_date(entry.get("updated"))
                    or _parse_date(entry.get("pubDate"))
                    or _parse_date(entry.get("published_parsed"))
                    or _parse_date(entry.get("updated_parsed"))
                )
                if pub_date and pub_date < cutoff_date:
                    continue

                article = _build_article(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", entry.get("description", "")),
                    published=pub_date,
                    source=feed_source,
                    url=entry.get("link", ""),
                )
                if article:
                    articles.append(article)
                    if len(articles) >= max_articles:
                        return articles
        except Exception as e:
            logger.warning(f"Failed to parse market RSS feed {feed_url}: {e}")

    return articles


def _fetch_market_news_from_newsapi(cutoff_date: datetime, max_articles: int) -> list[dict]:
    if not config.NEWS_API_KEY:
        return []

    params = {
        "country": config.NEWSAPI_COUNTRY,
        "category": "business",
        "pageSize": min(max_articles, config.NEWSAPI_PAGE_SIZE),
        "apiKey": config.NEWS_API_KEY,
    }

    try:
        response = requests.get(
            config.NEWSAPI_TOP_HEADLINES_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            logger.debug(
                "NewsAPI top headlines failed: status=%s body=%s",
                response.status_code,
                response.text[:300],
            )
            return []

        payload = response.json()
        feed_articles = payload.get("articles", []) if isinstance(payload, dict) else []
        articles = []
        for item in feed_articles:
            pub_date = _parse_date(item.get("publishedAt"))
            if pub_date and pub_date < cutoff_date:
                continue

            source_name = ""
            source_obj = item.get("source")
            if isinstance(source_obj, dict):
                source_name = source_obj.get("name", "")

            article = _build_article(
                title=item.get("title", ""),
                summary=item.get("description", "") or item.get("content", ""),
                published=pub_date,
                source=source_name or "NewsAPI",
                url=item.get("url", ""),
            )
            if article:
                articles.append(article)
        return articles[:max_articles]
    except Exception as e:
        logger.debug(f"NewsAPI market fetch failed: {e}")
        return []


# ─────────────────────────────────────────────
# Macro Context
# ─────────────────────────────────────────────

def get_macro_context() -> dict:
    """
    Fetch current Indian macro context:
    - Nifty 50 trend
    - USD/INR, Brent, India VIX, Gold, US10Y
    - sector index returns
    - optional NSE FII/DII cash flow snapshot
    - recent macro headlines
    """
    cached = cache.get("macro", "india_macro", config.CACHE_TTL_MACRO)
    if cached is not None:
        return cached

    macro: dict[str, Any] = {}

    try:
        nifty_hist = _get_history("^NSEI", period="3mo")
        if nifty_hist is not None and not nifty_hist.empty:
            macro["nifty50_current"] = _safe_float(nifty_hist["Close"].iloc[-1])
            macro["nifty50_1m_return"] = _compute_return(nifty_hist, bars_back=22)
            macro["nifty50_3m_return"] = _compute_return(nifty_hist, bars_back=len(nifty_hist) - 1)
    except Exception as e:
        logger.warning(f"Could not fetch Nifty 50 data: {e}")

    macro["usd_inr"] = _latest_price("INR=X")
    macro["brent_crude_usd"] = _latest_price("BZ=F")
    macro["india_vix"] = _latest_price("^INDIAVIX")
    macro["gold_usd_oz"] = _latest_price("GC=F")
    macro["us10y_yield"] = _latest_price("^TNX")

    sector_tickers = {
        "Nifty Bank": "^NSEBANK",
        "Nifty IT": "^CNXIT",
        "Nifty Pharma": "^CNXPHARMA",
        "Nifty Auto": "^CNXAUTO",
        "Nifty FMCG": "^CNXFMCG",
        "Nifty Metal": "^CNXMETAL",
        "Nifty Energy": "^CNXENERGY",
        "Nifty Realty": "^CNXREALTY",
    }
    sector_returns: dict[str, float] = {}
    for sector_name, ticker_sym in sector_tickers.items():
        try:
            hist = _get_history(ticker_sym, period="1mo")
            if hist is not None and not hist.empty and len(hist) > 1:
                ret = _compute_return(hist, bars_back=len(hist) - 1)
                if ret is not None:
                    sector_returns[sector_name] = ret
        except Exception:
            continue
    macro["sector_1m_returns"] = sector_returns

    fii_dii_snapshot = _fetch_fii_dii_flows()
    if fii_dii_snapshot:
        macro.update(fii_dii_snapshot)

    macro["macro_headlines"] = get_market_news(days=2)[:8]
    macro["fetch_time"] = datetime.now().isoformat()

    # Optional: attach latest NiftyIndices reports snapshot (cached).
    # Keep this fast: never let report scraping stall macro context.
    if getattr(config, "REPORTS_ENABLED", True):
        try:
            from data.reports import get_latest_reports_snapshot
            macro["niftyindices_reports"] = get_latest_reports_snapshot(force_refresh=False)
        except Exception:
            macro["niftyindices_reports"] = {"error": "reports_unavailable"}
    else:
        macro["niftyindices_reports"] = {"disabled": True}

    cache.set("macro", "india_macro", macro)
    return macro


def _fetch_fii_dii_flows() -> dict:
    payload = _fetch_nse_json(NSE_FII_DII_ENDPOINT)
    if not payload:
        return {}

    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}

    output: dict[str, Any] = {}
    trade_date = None
    for row in rows:
        if not isinstance(row, dict):
            continue

        category = (row.get("category") or row.get("clientType") or "").upper()
        if not trade_date:
            trade_date = row.get("date") or row.get("tradedDate") or row.get("tradeDate")

        buy_value = _parse_numeric(row.get("buyValue") or row.get("buy"))
        sell_value = _parse_numeric(row.get("sellValue") or row.get("sell"))
        net_value = _parse_numeric(row.get("netValue") or row.get("net"))

        if "FII" in category or "FPI" in category:
            output["fii_buy_cash_cr"] = buy_value
            output["fii_sell_cash_cr"] = sell_value
            output["fii_net_cash_cr"] = net_value
        elif "DII" in category:
            output["dii_buy_cash_cr"] = buy_value
            output["dii_sell_cash_cr"] = sell_value
            output["dii_net_cash_cr"] = net_value

    if trade_date:
        output["fii_dii_trade_date"] = str(trade_date)
    return output


def _fetch_nse_json(url: str) -> dict:
    session = _get_nse_session()
    try:
        session.get(NSE_BOOTSTRAP_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            logger.debug("NSE request failed status=%s url=%s", response.status_code, url)
            return {}
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except Exception as e:
        logger.debug(f"NSE fetch failed ({url}): {e}")
        return {}


def _get_nse_session() -> requests.Session:
    global _NSE_SESSION
    if _NSE_SESSION is None:
        _NSE_SESSION = requests.Session()
        _NSE_SESSION.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
            "Connection": "keep-alive",
        })
    return _NSE_SESSION


def _latest_price(symbol: str) -> Optional[float]:
    try:
        hist = _get_history(symbol, period="7d")
        if hist is not None and not hist.empty:
            return _safe_float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def _get_history(symbol: str, period: str):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period, interval="1d")
    except Exception:
        return None


def _compute_return(hist, bars_back: int) -> Optional[float]:
    try:
        if hist is None or hist.empty or bars_back <= 0 or len(hist) <= bars_back:
            return None
        current = _safe_float(hist["Close"].iloc[-1])
        prior = _safe_float(hist["Close"].iloc[-(bars_back + 1)])
        if current is None or prior in (None, 0):
            return None
        return (current - prior) / prior
    except Exception:
        return None


# ─────────────────────────────────────────────
# Scoring Helpers
# ─────────────────────────────────────────────

def compute_composite_score(fundamentals: dict) -> dict:
    """
    Compute a composite screening score for Stage 1.
    Returns scores on 0-100 scale.

    Score components:
    - Valuation (PE, PB): 25%
    - Quality (ROE, margins, debt): 25%
    - Growth (revenue, earnings): 25%
    - Momentum (52w return, 6m return): 25%
    """
    scores = {}

    pe = _safe_float(fundamentals.get("trailingPE"))
    pb = _safe_float(fundamentals.get("priceToBook"))
    pe_score = _score_pe(pe)
    pb_score = _score_pb(pb)
    scores["valuation"] = (pe_score * 0.6 + pb_score * 0.4)

    roe = _safe_float(fundamentals.get("returnOnEquity")) or 0
    margin = _safe_float(fundamentals.get("profitMargins")) or 0
    debt_eq = _safe_float(fundamentals.get("debtToEquity")) or 0
    roe_score = min(100, max(0, roe * 200))
    margin_score = min(100, max(0, margin * 500))
    debt_score = max(0, 100 - debt_eq * 2)
    scores["quality"] = (roe_score * 0.4 + margin_score * 0.3 + debt_score * 0.3)

    rev_growth = _safe_float(fundamentals.get("revenueGrowth")) or 0
    earn_growth = _safe_float(fundamentals.get("earningsGrowth")) or 0
    rev_score = min(100, max(0, rev_growth * 300))
    earn_score = min(100, max(0, earn_growth * 200))
    scores["growth"] = (rev_score * 0.5 + earn_score * 0.5)

    ret_52w = _safe_float(fundamentals.get("52WeekChange")) or _safe_float(fundamentals.get("52w_return_pct")) or 0
    ret_6m = _safe_float(fundamentals.get("6m_return_pct")) or 0
    mom_52w = min(100, max(0, (ret_52w + 0.3) * 200))
    mom_6m = min(100, max(0, (ret_6m + 0.2) * 250))
    scores["momentum"] = (mom_52w * 0.5 + mom_6m * 0.5)

    composite = (
        scores["valuation"] * 0.25
        + scores["quality"] * 0.25
        + scores["growth"] * 0.25
        + scores["momentum"] * 0.25
    )
    scores["composite"] = composite
    return scores


def _score_pe(pe: Optional[float]) -> float:
    """Score trailing PE: sweet spot 10-25, penalize very high or negative."""
    pe = _safe_float(pe)
    if pe is None or pe <= 0:
        return 30.0
    if pe <= 10:
        return 85.0
    if pe <= 20:
        return 100.0
    if pe <= 35:
        return 100.0 - (pe - 20) * 3
    if pe <= 60:
        return 55.0 - (pe - 35) * 1.5
    return 10.0


def _score_pb(pb: Optional[float]) -> float:
    """Score Price-to-Book: lower is generally better."""
    pb = _safe_float(pb)
    if pb is None or pb <= 0:
        return 30.0
    if pb <= 1:
        return 100.0
    if pb <= 3:
        return 100.0 - (pb - 1) * 15
    if pb <= 7:
        return 70.0 - (pb - 3) * 10
    return 20.0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _build_search_terms(symbol: str, company_name: str) -> list[str]:
    terms = [symbol.strip().upper(), company_name.strip()]
    clean = re.sub(r"[^a-zA-Z0-9 ]+", " ", company_name.lower())
    tokens = [w for w in clean.split() if len(w) >= 4 and w not in {
        "limited", "ltd", "india", "industries", "company", "corp", "corporation",
    }]
    if tokens:
        terms.extend(tokens[:3])
        if len(tokens) >= 2:
            terms.append(f"{tokens[0]} {tokens[1]}")
    return [t for t in dict.fromkeys(terms) if t]


def _load_rss_entries(feed_url: str) -> tuple[str, list[dict]]:
    if feedparser is not None:
        try:
            feed = feedparser.parse(feed_url)
            feed_source = (feed.feed or {}).get("title", feed_url.split("/")[2])
            entries: list[dict] = []
            for entry in feed.entries:
                entries.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", "")),
                    "description": entry.get("description", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "updated": entry.get("updated", ""),
                    "pubDate": entry.get("pubDate", ""),
                    "published_parsed": entry.get("published_parsed"),
                    "updated_parsed": entry.get("updated_parsed"),
                })
            if entries and not getattr(feed, "bozo", False):
                return feed_source, entries
            logger.debug(
                "RSS feedparser fallback for %s (entries=%s bozo=%s status=%s)",
                feed_url,
                len(entries),
                getattr(feed, "bozo", False),
                getattr(feed, "status", None),
            )
        except Exception as e:
            logger.debug("RSS feedparser parse failed for %s: %s", feed_url, e)

    _warn_missing_feedparser_once()
    return _load_rss_entries_stdlib(feed_url)


def _load_rss_entries_stdlib(feed_url: str) -> tuple[str, list[dict]]:
    response = requests.get(
        feed_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        },
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)

    channel = root.find("channel")
    if channel is not None:
        feed_source = _xml_text(channel.find("title")) or feed_url.split("/")[2]
        entries = []
        for item in channel.findall("item"):
            entries.append({
                "title": _xml_text(item.find("title")),
                "summary": _xml_text(item.find("description")),
                "description": _xml_text(item.find("description")),
                "link": _xml_text(item.find("link")),
                "published": _xml_text(item.find("pubDate")),
                "updated": _xml_text(item.find("updated")),
                "pubDate": _xml_text(item.find("pubDate")),
                "published_parsed": None,
                "updated_parsed": None,
            })
        return feed_source, entries

    feed_source = _xml_text(root.find(".//{*}title")) or feed_url.split("/")[2]
    entries = []
    for item in root.findall(".//{*}entry"):
        link_el = item.find("{*}link")
        link_val = ""
        if link_el is not None:
            link_val = link_el.attrib.get("href", "") or (link_el.text or "")
        entries.append({
            "title": _xml_text(item.find("{*}title")),
            "summary": _xml_text(item.find("{*}summary")) or _xml_text(item.find("{*}content")),
            "description": _xml_text(item.find("{*}summary")) or _xml_text(item.find("{*}content")),
            "link": link_val,
            "published": _xml_text(item.find("{*}published")) or _xml_text(item.find("{*}updated")),
            "updated": _xml_text(item.find("{*}updated")),
            "pubDate": "",
            "published_parsed": None,
            "updated_parsed": None,
        })
    return feed_source, entries


def _is_article_relevant(text: str, search_terms: list[str]) -> bool:
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text).lower()
    for term in search_terms:
        term_norm = term.strip().lower()
        if not term_norm:
            continue
        if len(term_norm) <= 4:
            if re.search(rf"\b{re.escape(term_norm)}\b", normalized):
                return True
        elif term_norm in normalized:
            return True
    return False


def _build_newsapi_stock_query(symbol: str, company_name: str) -> str:
    cleaned_company = re.sub(r"\s+", " ", company_name).strip()
    symbol_part = f"\"{symbol}\"" if len(symbol) <= 5 else symbol
    return f"({symbol_part} OR \"{cleaned_company}\") AND (India OR NSE OR BSE OR stock)"


def _google_news_rss_url(query: str) -> str:
    # `when:7d` is supported by Google News search, but we still enforce cutoff_date
    # in case Google returns older items.
    from urllib.parse import quote_plus

    q = quote_plus(query + " when:7d")
    return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def _build_article(
    title: str,
    summary: str,
    published: Optional[datetime],
    source: str,
    url: str,
) -> dict:
    clean_title = _normalize_whitespace(_clean_html(title))
    if not clean_title:
        return {}

    summary_limit = 250 if getattr(config, "CAVEMAN_MODE", False) else 500
    article = {
        "title": clean_title,
        "summary": _truncate(_normalize_whitespace(_clean_html(summary)), summary_limit),
        "published": published.isoformat() if isinstance(published, datetime) else "",
        "source": _normalize_whitespace(source) or "Unknown",
    }
    if not getattr(config, "CAVEMAN_MODE", False):
        article["url"] = (url or "").strip()
    return article


def _dedupe_sort_limit_articles(
    articles: list[dict],
    cutoff_date: datetime,
    max_articles: int,
) -> list[dict]:
    prepared: list[dict] = []
    for article in articles:
        title = _normalize_whitespace(article.get("title", ""))
        if not title:
            continue

        published_dt = _parse_date(article.get("published"))
        if published_dt and published_dt < cutoff_date:
            continue

        source = _normalize_whitespace(article.get("source", "Unknown"))
        summary = article.get("summary", "")
        url = article.get("url", "")
        prepared_entry = {
            "title": title,
            "summary": summary,
            "published": published_dt.isoformat() if published_dt else "",
            "source": source,
        }
        if not getattr(config, "CAVEMAN_MODE", False):
            prepared_entry["url"] = url
        prepared.append(prepared_entry)

    prepared.sort(key=lambda x: _parse_date(x.get("published")) or datetime.min, reverse=True)

    seen = set()
    unique = []
    for article in prepared:
        key = (
            _normalize_whitespace(article.get("title", "")).lower(),
            _normalize_whitespace(article.get("source", "")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
        if len(unique) >= max_articles:
            break
    return unique


def _parse_date(value: Any) -> Optional[datetime]:
    """Parse timestamps from strings, epoch seconds, datetime, or feedparser structs."""
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        return _to_naive_utc(value)

    if isinstance(value, (int, float)):
        try:
            if value <= 0:
                return None
            return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None

    if hasattr(value, "tm_year") and hasattr(value, "tm_mon"):
        try:
            return datetime(*value[:6])
        except Exception:
            return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(raw)
            return _to_naive_utc(dt)
        except ValueError:
            pass

        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S +0000",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                parsed = datetime.strptime(raw, fmt)
                return _to_naive_utc(parsed)
            except ValueError:
                continue

    return None


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _clean_html(text: str) -> str:
    if not text:
        return ""
    if BeautifulSoup is None:
        return re.sub(r"<[^>]+>", " ", text).strip()
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ").strip()
    except Exception:
        return re.sub(r"<[^>]+>", " ", text).strip()


def _truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else text[:max_len].rstrip() + "…"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _xml_text(element) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").strip()
        if not cleaned or cleaned == "-":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        # pandas may pass a single-element Series; normalize to scalar
        if hasattr(value, "iloc"):
            try:
                value = value.iloc[0]
            except Exception:
                pass
        return float(value)
    except (TypeError, ValueError):
        return None


def _warn_missing_feedparser_once() -> None:
    global _WARNED_MISSING_FEEDPARSER
    if not _WARNED_MISSING_FEEDPARSER:
        logger.warning("feedparser is not installed; using built-in RSS parser fallback. Install dependencies with `pip install -r requirements.txt` for fuller feed compatibility.")
        _WARNED_MISSING_FEEDPARSER = True


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing Data Fetcher")
    print("=" * 60)

    test_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    for sym in test_stocks:
        data = get_fundamentals(sym)
        score = compute_composite_score(data)
        print(f"\n{sym}:")
        print(f"  Price: ₹{data.get('price', 'N/A')}")
        print(f"  PE: {data.get('trailingPE', 'N/A')}")
        print(f"  ROE: {data.get('returnOnEquity', 'N/A')}")
        print(f"  Composite Score: {score['composite']:.1f}/100")

    print("\n\nTesting News Fetch (RELIANCE, last 7 days):")
    news = get_news_for_stock("RELIANCE", "Reliance Industries", days=7)
    print(f"Found {len(news)} articles")
    for article in news[:3]:
        print(f"  [{article['source']}] {article['title'][:80]}")

    print("\n\nMacro Context:")
    macro = get_macro_context()
    print(f"  Nifty 50: {macro.get('nifty50_current', 'N/A')}")
    print(f"  USD/INR: {macro.get('usd_inr', 'N/A')}")
    print(f"  India VIX: {macro.get('india_vix', 'N/A')}")
    print(f"  Sector Returns: {macro.get('sector_1m_returns', {})}")
