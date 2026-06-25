#!/usr/bin/env python3
"""
Token simulation for:
  1) initial portfolio bootstrap run
  2) 5 daily-lite runs (Mon–Fri)

This is an estimate based on:
  - last successful Stage2/Stage3 token usage (per-agent/per-stock)
  - an optional Stage4 baseline (can be overridden via env)

It does NOT attempt to price USD costs (provider-specific).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass
class Baselines:
    bull_tokens: int
    bear_tokens: int
    synth_tokens: int
    stage3_tokens: int
    stage4_tokens: int


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _latest_run_dirs() -> list[str]:
    dirs = sorted(glob.glob("storage/pipeline_runs/*"))
    return [d for d in dirs if os.path.isdir(d)]


def _latest_successful_stage2_run_dir() -> str | None:
    # Prefer latest run that has stage2_output with at least one successful bull+bear.
    for d in reversed(_latest_run_dirs()):
        p = os.path.join(d, "stage2_output.json")
        if not os.path.exists(p):
            continue
        try:
            data = _load_json(p)
            research = data.get("research") or {}
            for _, rec in research.items():
                if rec.get("bull_success") and rec.get("bear_success"):
                    return d
        except Exception:
            continue
    return None


def _latest_successful_stage3_run_dir() -> str | None:
    for d in reversed(_latest_run_dirs()):
        p = os.path.join(d, "stage3_output.json")
        if not os.path.exists(p):
            continue
        try:
            data = _load_json(p)
            if (data.get("total_modeled") or 0) > 0:
                return d
        except Exception:
            continue
    return None


def _estimate_stage4_baseline_tokens() -> int:
    """
    Stage4 isn't run in daily-lite. For bootstrap runs, Stage4 runs once.
    If you have a recent stage4_output.json, use its token_usage. Otherwise
    fall back to env override or a conservative default.
    """
    env = os.getenv("SIM_STAGE4_TOKENS", "").strip()
    if env:
        try:
            return int(float(env))
        except Exception:
            pass

    # Look for any stage4 outputs
    vals = []
    for d in _latest_run_dirs():
        p = os.path.join(d, "stage4_output.json")
        if not os.path.exists(p):
            continue
        try:
            data = _load_json(p)
            u = (data.get("token_usage") or {})
            vals.append(int(u.get("total_tokens") or 0))
        except Exception:
            continue
    if vals:
        return int(median(vals))

    # Default based on a measured Gemma run in this repo (~1553 tokens).
    return 1553


def baselines_from_latest_runs() -> Baselines:
    stage2_dir = _latest_successful_stage2_run_dir()
    if not stage2_dir:
        raise SystemExit("No successful Stage2 run found. Run `python3 run.py --mode daily` once.")

    s2 = _load_json(os.path.join(stage2_dir, "stage2_output.json"))
    # Take the first symbol with bull+bear success
    rec = None
    for _, r in (s2.get("research") or {}).items():
        if r.get("bull_success") and r.get("bear_success"):
            rec = r
            break
    if not rec:
        raise SystemExit("Stage2 output exists but has no successful bull+bear record.")

    bull_cases = rec.get("bull_cases") or []
    bear_cases = rec.get("bear_cases") or []
    bull_ok = [c for c in bull_cases if c.get("success") and (c.get("token_usage") or {}).get("total_tokens")]
    bear_ok = [c for c in bear_cases if c.get("success") and (c.get("token_usage") or {}).get("total_tokens")]
    if not bull_ok or not bear_ok:
        raise SystemExit("Stage2 record missing token usage for bull/bear.")

    bull_tokens = int(median([(c.get("token_usage") or {}).get("total_tokens", 0) for c in bull_ok]))
    bear_tokens = int(median([(c.get("token_usage") or {}).get("total_tokens", 0) for c in bear_ok]))

    synth_u = rec.get("synthesis_token_usage") or {}
    synth_tokens = int(synth_u.get("total_tokens") or 0)

    stage3_dir = _latest_successful_stage3_run_dir()
    if not stage3_dir:
        raise SystemExit("No successful Stage3 run found. Run `python3 run.py --mode daily` with at least 1 modeled stock.")
    s3 = _load_json(os.path.join(stage3_dir, "stage3_output.json"))
    models = s3.get("scenario_models") or []
    tokens = []
    for m in models:
        u = (m or {}).get("_token_usage") or {}
        t = int(u.get("total_tokens") or 0)
        if t:
            tokens.append(t)
    if not tokens:
        raise SystemExit("Stage3 output has no per-model token usage.")
    stage3_tokens = int(median(tokens))

    stage4_tokens = _estimate_stage4_baseline_tokens()

    return Baselines(
        bull_tokens=bull_tokens,
        bear_tokens=bear_tokens,
        synth_tokens=synth_tokens,
        stage3_tokens=stage3_tokens,
        stage4_tokens=stage4_tokens,
    )


def simulate(
    *,
    init_top_n: int,
    init_bull: int,
    init_bear: int,
    daily_days: int,
    daily_stocks_worst: int,
    daily_stocks_expected: int,
    b: Baselines,
) -> dict[str, Any]:
    per_stock_stage2 = init_bull * b.bull_tokens + init_bear * b.bear_tokens + b.synth_tokens
    per_stock_stage3 = b.stage3_tokens

    bootstrap = {
        "stage2_tokens": init_top_n * per_stock_stage2,
        "stage3_tokens": init_top_n * per_stock_stage3,
        "stage4_tokens": b.stage4_tokens,
        "stage5_tokens": 0,  # initial run uses the non-LLM first-run path
    }
    bootstrap["total_tokens"] = sum(bootstrap.values())

    daily_per_stock_stage2 = 1 * b.bull_tokens + 1 * b.bear_tokens + b.synth_tokens
    daily_per_stock_stage3 = b.stage3_tokens

    daily_worst_one_day = daily_stocks_worst * (daily_per_stock_stage2 + daily_per_stock_stage3)
    daily_expected_one_day = daily_stocks_expected * (daily_per_stock_stage2 + daily_per_stock_stage3)

    daily = {
        "one_day_worst": daily_worst_one_day,
        "one_day_expected": daily_expected_one_day,
        "five_days_worst": daily_days * daily_worst_one_day,
        "five_days_expected": daily_days * daily_expected_one_day,
    }

    totals = {
        "bootstrap_plus_week_expected": bootstrap["total_tokens"] + daily["five_days_expected"],
        "bootstrap_plus_week_worst": bootstrap["total_tokens"] + daily["five_days_worst"],
    }

    return {
        "baselines": b.__dict__,
        "bootstrap_assumptions": {"init_top_n": init_top_n, "init_bull": init_bull, "init_bear": init_bear},
        "daily_assumptions": {
            "days": daily_days,
            "stocks_expected": daily_stocks_expected,
            "stocks_worst": daily_stocks_worst,
            "daily_lite_config": "expected=~3 stocks/day, worst=5 (3 new + 2 flagged)",
        },
        "bootstrap_tokens": bootstrap,
        "daily_tokens": daily,
        "totals": totals,
    }


def main() -> None:
    # Defaults tuned for a <$20/month approach:
    # - bootstrap: top 20, 2 bull + 2 bear per stock (still heavy but manageable)
    # - daily-lite: expected ~3 stocks/day, worst-case 5 stocks/day
    init_top_n = int(os.getenv("SIM_INIT_TOP_N", "20"))
    init_bull = int(os.getenv("SIM_INIT_BULL", "2"))
    init_bear = int(os.getenv("SIM_INIT_BEAR", "2"))
    daily_days = int(os.getenv("SIM_DAYS", "5"))
    daily_worst = int(os.getenv("SIM_DAILY_WORST_STOCKS", "5"))
    daily_expected = int(os.getenv("SIM_DAILY_EXPECTED_STOCKS", "3"))

    b = baselines_from_latest_runs()
    out = simulate(
        init_top_n=init_top_n,
        init_bull=init_bull,
        init_bear=init_bear,
        daily_days=daily_days,
        daily_stocks_worst=daily_worst,
        daily_stocks_expected=daily_expected,
        b=b,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

