"""
Backtest harness: replay historical Stage 3 outputs against realized forward
returns to score the debate-driven EV calibration.

Method (no LLM rerun — uses already-written run artifacts):
  1. Walk storage/pipeline_runs/*/stage3_output.json.
  2. For each scenario_models entry, read:
       symbol, run_time (entry date), entry price,
       scenarios {bull, base, bear}.{probability, return_12m},
       probability_weighted_return_12m (EV),
       debate_scores (P_bull, P_base, P_bear if present).
  3. Pull yfinance history at run_date and run_date+horizon (default 365d, or
     min(today - run_date, 365d) for incomplete windows).
  4. Compute realized return, residual (realized - EV), and Brier score for the
     {bull/base/bear} probability distribution against the realized bucket
     (bucket boundaries = midpoints of scenario returns).
  5. Aggregate: per-run + global metrics. Persist to storage/backtests/<run>.json.

Output metrics:
  - mean_residual, mean_abs_residual: bias + miss
  - hit_rate: fraction where EV sign matches realized sign
  - rank_corr (Spearman) EV vs realized: alpha indicator
  - mean_brier (lower better): scenario probability calibration
  - sample size

Free sources only (yfinance).
"""

from pathlib import Path
from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import datetime, timedelta
from typing import Any, Optional


import config
import data.cache as cache

logger = logging.getLogger(__name__)

_PRICE_CACHE_TTL = 7 * 24 * 3600  # 7-day cache for historical bars


def _parse_run_time(run_meta: dict, fallback_dir_name: str) -> Optional[datetime]:
    val = run_meta.get("run_time") or run_meta.get("start_time")
    if val:
        try:
            return datetime.fromisoformat(val.replace("Z", ""))
        except Exception:
            pass
    # Fallback: directory name is YYYYMMDD_HHMMSS
    try:
        return datetime.strptime(fallback_dir_name, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def _yf_close(yf_symbol: str, target_date: datetime, window_days: int = 7) -> Optional[float]:
    """Closing price near target_date (±window). Cached. None if unavailable."""
    key = f"{yf_symbol}:{target_date.date().isoformat()}:{window_days}"
    cached = cache.get("backtest_close", key, ttl=_PRICE_CACHE_TTL)
    if cached is not None:
        return cached if cached != "MISS" else None

    try:
        import yfinance as yf
        start = (target_date - timedelta(days=window_days)).date()
        end = (target_date + timedelta(days=window_days)).date()
        hist = yf.Ticker(yf_symbol).history(start=start.isoformat(), end=end.isoformat(), auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            cache.set("backtest_close", key, "MISS")
            return None
        # Use closest available trading day on/after target_date, else closest before.
        dt_idx = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
        target = target_date.replace(tzinfo=None)
        after = [(d, hist.iloc[i]["Close"]) for i, d in enumerate(dt_idx) if d >= target]
        if after:
            price = float(after[0][1])
        else:
            price = float(hist.iloc[-1]["Close"])
        cache.set("backtest_close", key, price)
        return price
    except Exception as e:
        logger.debug("yfinance history failed %s %s: %s", yf_symbol, target_date, e)
        cache.set("backtest_close", key, "MISS")
        return None


def _scenario_returns(scenarios: dict) -> Optional[tuple[float, float, float]]:
    try:
        b = scenarios.get("bull", {}) or {}
        m = scenarios.get("base", {}) or {}
        r = scenarios.get("bear", {}) or {}
        return (
            float(b.get("return_12m") or b.get("target") or 0.0),
            float(m.get("return_12m") or m.get("target") or 0.0),
            float(r.get("return_12m") or r.get("target") or 0.0),
        )
    except Exception:
        return None


def _brier_score(probs: tuple[float, float, float], realized_idx: int) -> float:
    """Multiclass Brier: sum((p_i - y_i)^2). y_i = 1 for realized bucket, else 0."""
    target = [0.0, 0.0, 0.0]
    target[realized_idx] = 1.0
    return sum((p - t) ** 2 for p, t in zip(probs, target))


def _bucket_realized(realized: float, bull_r: float, base_r: float, bear_r: float) -> int:
    """Return idx 0=bull, 1=base, 2=bear based on which scenario realized closest exceeds."""
    # Use midpoints between bull/base and base/bear as cutoffs.
    bull_base_mid = (bull_r + base_r) / 2.0
    base_bear_mid = (base_r + bear_r) / 2.0
    if realized >= bull_base_mid:
        return 0  # bull
    if realized >= base_bear_mid:
        return 1  # base
    return 2  # bear


def _spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    def _rank(vals: list[float]) -> list[float]:
        idx = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[idx[j + 1]] == vals[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks
    rx, ry = _rank(xs), _rank(ys)
    mx = statistics.mean(rx); my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def backtest_run(run_dir: Path, horizon_days: int = 365) -> Optional[dict]:
    stage3 = run_dir / "stage3_output.json"
    if not stage3.exists():
        return None

    try:
        data = json.loads(stage3.read_text())
    except Exception as e:
        logger.warning("Failed to read %s: %s", stage3, e)
        return None

    run_meta_path = run_dir / "run_summary.json"
    run_meta = {}
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text())
        except Exception:
            pass

    run_time = _parse_run_time({**data, **run_meta}, run_dir.name)
    if not run_time:
        logger.warning("Could not determine run_time for %s", run_dir)
        return None

    now = datetime.now()
    elapsed_days = (now - run_time).days
    if elapsed_days < 30:
        logger.info("Skipping %s — only %d days elapsed (min 30)", run_dir.name, elapsed_days)
        return None
    effective_horizon = min(horizon_days, elapsed_days)
    target_date = run_time + timedelta(days=effective_horizon)

    rows = []
    evs: list[float] = []
    realizeds: list[float] = []
    briers: list[float] = []
    hits = 0
    bucket_counts = [0, 0, 0]

    for sm in data.get("scenario_models", []):
        if sm.get("error"):
            continue
        symbol = (sm.get("symbol") or "").upper()
        if not symbol:
            continue
        yf_symbol = f"{symbol}.NS"

        entry_price = sm.get("current_price") or sm.get("price")
        try:
            entry_price = float(entry_price) if entry_price is not None else None
        except Exception:
            entry_price = None
        if not entry_price:
            entry_price = _yf_close(yf_symbol, run_time)
        if not entry_price:
            continue

        exit_price = _yf_close(yf_symbol, target_date)
        if not exit_price:
            continue

        realized = (exit_price - entry_price) / entry_price
        ev = sm.get("probability_weighted_return_12m")
        try:
            ev = float(ev) if ev is not None else None
        except Exception:
            ev = None

        scen = sm.get("scenarios") or {}
        sret = _scenario_returns(scen)
        probs = None
        brier = None
        bucket = None
        if sret:
            bull_r, base_r, bear_r = sret
            p_bull = scen.get("bull", {}).get("probability") or sm.get("debate_scores", {}).get("P_bull")
            p_base = scen.get("base", {}).get("probability") or sm.get("debate_scores", {}).get("P_base")
            p_bear = scen.get("bear", {}).get("probability") or sm.get("debate_scores", {}).get("P_bear")
            try:
                p_bull = float(p_bull or 0)
                p_base = float(p_base or 0)
                p_bear = float(p_bear or 0)
                total = p_bull + p_base + p_bear
                if total > 0:
                    probs = (p_bull / total, p_base / total, p_bear / total)
                    bucket = _bucket_realized(realized, bull_r, base_r, bear_r)
                    brier = _brier_score(probs, bucket)
                    briers.append(brier)
                    bucket_counts[bucket] += 1
            except Exception:
                pass

        rows.append({
            "symbol": symbol,
            "sector": sm.get("sector"),
            "entry_date": run_time.date().isoformat(),
            "exit_date": target_date.date().isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_return": round(realized, 4),
            "ev_predicted": round(ev, 4) if ev is not None else None,
            "residual": round(realized - ev, 4) if ev is not None else None,
            "probs_bull_base_bear": probs,
            "realized_bucket": ["bull", "base", "bear"][bucket] if bucket is not None else None,
            "brier_score": round(brier, 4) if brier is not None else None,
            "recommendation": sm.get("recommendation"),
        })
        if ev is not None:
            evs.append(ev)
            realizeds.append(realized)
            if (ev >= 0) == (realized >= 0):
                hits += 1

    n = len(rows)
    if n == 0:
        logger.info("Backtest %s: no usable rows", run_dir.name)
        return None

    metrics = {
        "n": n,
        "horizon_days": effective_horizon,
        "mean_residual": round(statistics.mean(r["residual"] for r in rows if r["residual"] is not None), 4)
            if any(r["residual"] is not None for r in rows) else None,
        "mean_abs_residual": round(statistics.mean(abs(r["residual"]) for r in rows if r["residual"] is not None), 4)
            if any(r["residual"] is not None for r in rows) else None,
        "hit_rate": round(hits / len(evs), 3) if evs else None,
        "rank_corr_ev_vs_realized": round(_spearman(evs, realizeds), 3) if _spearman(evs, realizeds) is not None else None,
        "mean_brier": round(statistics.mean(briers), 4) if briers else None,
        "realized_bucket_counts": {"bull": bucket_counts[0], "base": bucket_counts[1], "bear": bucket_counts[2]},
        "mean_realized": round(statistics.mean(realizeds), 4) if realizeds else None,
        "mean_ev_predicted": round(statistics.mean(evs), 4) if evs else None,
    }

    return {
        "run_id": run_dir.name,
        "run_time": run_time.isoformat(),
        "target_date": target_date.isoformat(),
        "metrics": metrics,
        "positions": rows,
    }


def run_backtest(horizon_days: int = 365) -> dict:
    """Backtest all historical pipeline runs. Returns aggregate summary."""
    runs_dir = config.STORAGE_DIR / "pipeline_runs"
    out_dir = config.STORAGE_DIR / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for run_dir in sorted(runs_dir.glob("*/")):
        result = backtest_run(run_dir, horizon_days=horizon_days)
        if result is None:
            continue
        out_path = out_dir / f"{run_dir.name}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info("Backtest %s → %s (n=%d, hit=%s, corr=%s, brier=%s)",
                    run_dir.name, out_path,
                    result["metrics"]["n"],
                    result["metrics"]["hit_rate"],
                    result["metrics"]["rank_corr_ev_vs_realized"],
                    result["metrics"]["mean_brier"])
        all_results.append(result)

    # Aggregate across runs
    agg = {
        "runs_backtested": len(all_results),
        "total_positions": sum(r["metrics"]["n"] for r in all_results),
    }
    if all_results:
        flat_residuals = [
            p["residual"] for r in all_results for p in r["positions"] if p["residual"] is not None
        ]
        flat_evs = [
            p["ev_predicted"] for r in all_results for p in r["positions"] if p["ev_predicted"] is not None
        ]
        flat_realized = [
            p["realized_return"] for r in all_results for p in r["positions"]
            if p["realized_return"] is not None and p["ev_predicted"] is not None
        ]
        flat_briers = [
            p["brier_score"] for r in all_results for p in r["positions"] if p["brier_score"] is not None
        ]
        hits = sum(1 for r in all_results for p in r["positions"]
                   if p["ev_predicted"] is not None and (p["ev_predicted"] >= 0) == (p["realized_return"] >= 0))
        agg.update({
            "global_mean_residual": round(statistics.mean(flat_residuals), 4) if flat_residuals else None,
            "global_mean_abs_residual": round(statistics.mean(abs(x) for x in flat_residuals), 4) if flat_residuals else None,
            "global_hit_rate": round(hits / len(flat_evs), 3) if flat_evs else None,
            "global_rank_corr": round(_spearman(flat_evs, flat_realized), 3)
                if _spearman(flat_evs, flat_realized) is not None else None,
            "global_mean_brier": round(statistics.mean(flat_briers), 4) if flat_briers else None,
            "global_mean_realized": round(statistics.mean(flat_realized), 4) if flat_realized else None,
            "global_mean_ev": round(statistics.mean(flat_evs), 4) if flat_evs else None,
        })

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(agg, indent=2, default=str))
    logger.info("Aggregate backtest summary saved to %s", summary_path)
    logger.info("AGGREGATE: %s", json.dumps(agg, default=str))
    return agg
