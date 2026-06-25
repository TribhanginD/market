"""
Sentiment quant layer — FinBERT-based sentiment scoring for NSE stocks.

Flow:
  1. Fetch recent news headlines via existing fetcher (cached)
  2. Score with ProsusAI/finbert via HuggingFace InferenceClient
  3. Return net sentiment score per symbol: -1.0 (bearish) to +1.0 (bullish)

Requires HF_TOKEN in .env.
Falls back to neutral (0.0) if API unavailable or no headlines found.
"""

import logging


import config
from data import cache as _cache_mod

logger = logging.getLogger(__name__)

HF_TOKEN = getattr(config, "HF_TOKEN", "") or __import__("os").getenv("HF_TOKEN", "")
FINBERT_MODEL = "ProsusAI/finbert"
CACHE_TTL_SENTIMENT = 6 * 3600  # 6 hours

_client = None          # lazy InferenceClient
_hf_unavailable = False # circuit-breaker: set True on first unrecoverable failure


def _get_client():
    global _client
    if _client is None:
        from huggingface_hub import InferenceClient
        _client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN,
        )
    return _client


def _hf_score_batch(texts: list[str]) -> list[dict]:
    """
    Score a batch of texts with FinBERT via huggingface_hub InferenceClient.
    Returns list of {"label": str, "score": float} per text (top label only).
    Returns [] on failure.
    """
    global _hf_unavailable
    if not texts or _hf_unavailable:
        return []

    results = []
    for text in texts:
        try:
            preds = _get_client().text_classification(text, model=FINBERT_MODEL)
            # preds is list of ClassificationOutput; pick highest score
            if preds:
                best = max(preds, key=lambda p: p.score)
                results.append({"label": best.label.lower(), "score": float(best.score)})
            else:
                results.append(None)
        except Exception as e:
            err = str(e)
            if any(k in err for k in ("NameResolution", "Failed to resolve", "nodename", "Connection", "timeout")):
                _hf_unavailable = True
                logger.warning("FinBERT unreachable — disabling for this session: %s", err[:120])
                results.append(None)
                break
            logger.warning("FinBERT score failed for text: %s", err[:120])
            results.append(None)
    return results


def _net_sentiment(scored: list) -> float:
    """
    Aggregate FinBERT results → single score in [-1, 1].
    positive → +score, negative → -score, neutral → 0.
    """
    total = 0.0
    count = 0
    for item in scored:
        if not isinstance(item, dict):
            continue
        label = item.get("label", "neutral")
        score = float(item.get("score", 0))
        if label == "positive":
            total += score
        elif label == "negative":
            total -= score
        count += 1
    return round(total / count, 4) if count else 0.0


def get_sentiment_score(symbol: str, company_name: str, days: int = 3) -> dict:
    """
    Return sentiment for a symbol.
    Returns {"symbol", "sentiment_score" (-1..1), "headline_count", "source"}.
    """
    cache_key = f"sentiment_{symbol}_{days}d"
    cached = _cache_mod.get("sentiment", cache_key, CACHE_TTL_SENTIMENT)
    if cached is not None:
        return cached

    try:
        from data.fetcher import get_news_for_stock
        articles = get_news_for_stock(symbol, company_name, days=days, max_articles=15)
    except Exception as e:
        logger.warning("News fetch failed for %s: %s", symbol, e)
        articles = []

    headlines = [a.get("title", "") for a in articles if a.get("title")]
    if not headlines:
        result = {"symbol": symbol, "sentiment_score": 0.0, "headline_count": 0, "source": "neutral_fallback"}
        _cache_mod.set("sentiment", cache_key, result)
        return result

    scored = _hf_score_batch(headlines)
    score = _net_sentiment(scored)

    result = {
        "symbol": symbol,
        "sentiment_score": score,
        "headline_count": len(headlines),
        "source": "finbert" if any(s is not None for s in scored) else "neutral_fallback",
    }
    _cache_mod.set("sentiment", cache_key, result)
    return result


def batch_sentiment_scores(
    stocks: list[dict],
    days: int = 3,
    skip_if_no_token: bool = True,
) -> dict[str, float | None]:
    """
    Score batch of stocks. Returns symbol → sentiment_score map.
    Values are float in [-1, 1], or None when sentiment is unavailable
    (circuit open, HF_TOKEN missing, or per-symbol failure).
    Callers must treat None as missing data, not neutral.
    """
    if skip_if_no_token and not HF_TOKEN:
        logger.info("HF_TOKEN not set — skipping FinBERT (sentiment unavailable)")
        return {s["symbol"]: None for s in stocks}

    scores: dict[str, float | None] = {}
    skipped = 0
    for stock in stocks:
        sym = stock.get("symbol", "")
        name = stock.get("company_name", sym)
        if _hf_unavailable:
            scores[sym] = None
            skipped += 1
            continue
        try:
            result = get_sentiment_score(sym, name, days=days)
            scores[sym] = result["sentiment_score"]
        except Exception as e:
            logger.warning("Sentiment score failed for %s: %s", sym, e)
            scores[sym] = None
    if skipped:
        logger.warning("FinBERT circuit open — sentiment unavailable for %d symbols", skipped)
    return scores
