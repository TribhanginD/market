"""
Broker actions + earnings revisions feed for NSE stocks.

Free sources only:
- yfinance `recommendations` (analyst rec counts over last 4 months) → derive net
  rating revision (broker upgrade/downgrade pressure).
- yfinance `analyst_price_targets` → mean target + implied upside vs current.
- yfinance `eps_trend` (EPS estimate per period at 0/7/30/60/90 days ago) →
  earnings revisions (the single highest-alpha equity signal).
- yfinance `earnings_estimate` → forward EPS estimate baseline.

All results disk-cached (24h TTL by default) to avoid rate limits.
NSE coverage in yfinance is partial for some endpoints; functions degrade
gracefully and return empty dicts on missing data.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import data.cache as cache

logger = logging.getLogger(__name__)

# 24h TTL — broker/EPS revisions are slow signals; daily refresh is enough.
_TTL_SECONDS = 24 * 3600


def _yf_ticker(yf_symbol: str):
    import yfinance as yf
    return yf.Ticker(yf_symbol)


def _safe_df(getter):
    try:
        df = getter()
        if df is None:
            return None
        if hasattr(df, "empty") and df.empty:
            return None
        return df
    except Exception as e:
        logger.debug("yfinance call failed: %s", e)
        return None


def get_broker_actions(yf_symbol: str, current_price: Optional[float] = None) -> dict[str, Any]:
    """
    Returns broker pressure signal:
      {
        "rating_counts": [{"period": "0m", "strong_buy": N, "buy": N, "hold": N, "sell": N, "strong_sell": N}, ...],
        "net_rating_revision": float,   # (buy_now - buy_3m_ago) - (sell_now - sell_3m_ago); positive = upgrades dominate
        "price_target_mean": float,
        "price_target_median": float,
        "upside_pct": float | None,     # (mean_target / current_price - 1) * 100
        "num_targets": int,
      }
    Empty dict if yfinance returns no data for this ticker.
    """
    cached = cache.get("broker_actions", yf_symbol, ttl=_TTL_SECONDS)
    if cached is not None:
        return cached

    t = _yf_ticker(yf_symbol)
    out: dict[str, Any] = {}

    rec = _safe_df(lambda: t.recommendations)
    if rec is not None:
        rows = rec.to_dict(orient="records")
        cleaned = []
        for r in rows:
            cleaned.append({
                "period": r.get("period"),
                "strong_buy": int(r.get("strongBuy") or 0),
                "buy": int(r.get("buy") or 0),
                "hold": int(r.get("hold") or 0),
                "sell": int(r.get("sell") or 0),
                "strong_sell": int(r.get("strongSell") or 0),
            })
        out["rating_counts"] = cleaned

        # Compute net rating revision: positive = analysts moved toward buy
        by_period = {r["period"]: r for r in cleaned}
        now = by_period.get("0m")
        prior = by_period.get("-3m") or by_period.get("-2m") or by_period.get("-1m")
        if now and prior:
            buy_now = now["strong_buy"] + now["buy"]
            buy_prior = prior["strong_buy"] + prior["buy"]
            sell_now = now["sell"] + now["strong_sell"]
            sell_prior = prior["sell"] + prior["strong_sell"]
            out["net_rating_revision"] = float((buy_now - buy_prior) - (sell_now - sell_prior))

    targets = None
    try:
        targets = t.analyst_price_targets
    except Exception as e:
        logger.debug("analyst_price_targets failed for %s: %s", yf_symbol, e)
    if isinstance(targets, dict) and targets:
        mean_t = targets.get("mean")
        median_t = targets.get("median")
        out["price_target_mean"] = mean_t
        out["price_target_median"] = median_t
        out["num_targets"] = (
            (targets.get("numberOfAnalystOpinions") or targets.get("numAnalysts"))
            if isinstance(targets, dict) else None
        )
        if current_price and mean_t:
            try:
                out["upside_pct"] = round((float(mean_t) / float(current_price) - 1.0) * 100.0, 2)
            except Exception:
                pass

    cache.set("broker_actions", yf_symbol, out)
    return out


def get_earnings_revisions(yf_symbol: str) -> dict[str, Any]:
    """
    Returns earnings revision signal:
      {
        "eps_trend": [
          {"period": "0q", "current": x, "7d_ago": y, "30d_ago": y, "60d_ago": y, "90d_ago": y,
           "pct_revision_30d": +/-N, "pct_revision_90d": +/-N},
          ...
        ],
        "fwd_eps_estimate": float,
        "fwd_eps_growth": float,    # next year vs trailing year, as fraction
        "net_revision_30d": float,  # average pct revision across periods over 30d
        "net_revision_90d": float,
      }
    """
    cached = cache.get("earnings_revisions", yf_symbol, ttl=_TTL_SECONDS)
    if cached is not None:
        return cached

    t = _yf_ticker(yf_symbol)
    out: dict[str, Any] = {}

    trend = _safe_df(lambda: t.eps_trend)
    if trend is not None:
        # eps_trend has period as index
        periods: list[dict] = []
        revisions_30 = []
        revisions_90 = []
        for period, row in trend.iterrows():
            cur = row.get("current")
            d7 = row.get("7daysAgo")
            d30 = row.get("30daysAgo")
            d60 = row.get("60daysAgo")
            d90 = row.get("90daysAgo")

            def _pct(now, then):
                try:
                    if now is None or then is None:
                        return None
                    now_f = float(now); then_f = float(then)
                    if then_f == 0:
                        return None
                    return round((now_f - then_f) / abs(then_f) * 100.0, 2)
                except Exception:
                    return None

            r30 = _pct(cur, d30)
            r90 = _pct(cur, d90)
            if r30 is not None:
                revisions_30.append(r30)
            if r90 is not None:
                revisions_90.append(r90)
            periods.append({
                "period": str(period),
                "current": cur,
                "7d_ago": d7,
                "30d_ago": d30,
                "60d_ago": d60,
                "90d_ago": d90,
                "pct_revision_30d": r30,
                "pct_revision_90d": r90,
            })
        out["eps_trend"] = periods
        if revisions_30:
            out["net_revision_30d"] = round(sum(revisions_30) / len(revisions_30), 2)
        if revisions_90:
            out["net_revision_90d"] = round(sum(revisions_90) / len(revisions_90), 2)

    est = _safe_df(lambda: t.earnings_estimate)
    if est is not None:
        try:
            # Forward year row
            if "+1y" in est.index:
                row = est.loc["+1y"]
                out["fwd_eps_estimate"] = row.get("avg")
                out["fwd_eps_growth"] = row.get("growth")
        except Exception:
            pass

    cache.set("earnings_revisions", yf_symbol, out)
    return out


def get_research_signals(yf_symbol: str, current_price: Optional[float] = None) -> dict[str, Any]:
    """One-shot wrapper used by Stage 2 prefetch."""
    return {
        "broker_actions": get_broker_actions(yf_symbol, current_price=current_price),
        "earnings_revisions": get_earnings_revisions(yf_symbol),
    }
