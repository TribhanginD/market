#!/usr/bin/env python3
"""
Run real executions against an OpenAI-compatible endpoint to measure
actual tokens from run_summary.json outputs.

Runs:
  1) bootstrap: stages 1-5
  2) daily-lite: 5 runs (Mon-Fri simulation)
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "storage" / "pipeline_runs"


@dataclass
class RunTotals:
    run_id: str
    input_tokens: int
    output_tokens: int


def _load_summary(run_id: str) -> dict[str, Any]:
    p = RUNS_DIR / run_id / "run_summary.json"
    return json.loads(p.read_text())


def _latest_run_ids(n: int = 20) -> list[str]:
    dirs = sorted([p.name for p in RUNS_DIR.glob("*") if p.is_dir()])
    return dirs[-n:]


def _token_totals_from_summary(s: dict[str, Any]) -> tuple[int, int]:
    tt = s.get("total_tokens") or {}
    return int(tt.get("input") or 0), int(tt.get("output") or 0)


def _run(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    if load_dotenv:
        load_dotenv(dotenv_path=str(ROOT / ".env"))
    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "").strip()
    if not base_url:
        raise SystemExit("OPENAI_COMPAT_BASE_URL is not set in environment/.env")

    raw_model = (
        os.getenv("QWEN_MODEL")
        or os.getenv("OPENAI_COMPAT_MODEL")
        or "Qwen/Qwen2.5-7B-Instruct"
    ).strip()
    model = raw_model if raw_model.startswith("openai:") else f"openai:{raw_model}"

    # Tunables (override via env)
    bootstrap_top_n = int(os.getenv("BOOTSTRAP_TOP_N", "15"))
    daily_new = int(os.getenv("DAILY_LITE_NEW_CANDIDATES", "3"))
    daily_flagged = int(os.getenv("DAILY_LITE_FLAGGED_HOLDINGS", "2"))

    env = os.environ.copy()
    env.update(
        {
            "MODEL_FAST": model,
            "MODEL_SMART": model,
            "STAGE2_MODEL": model,
            "STAGE3_MODEL": model,
            "STAGE4_MODEL": model,
            "STAGE5_MODEL": model,
            "STAGE1_USE_LLM_REVIEW": "false",
            "STAGE2_BULL_AGENTS_PER_STOCK": "1",
            "STAGE2_BEAR_AGENTS_PER_STOCK": "1",
            "STAGE2_TOP_N": str(bootstrap_top_n),
            "DAILY_LITE_NEW_CANDIDATES": str(daily_new),
            "DAILY_LITE_FLAGGED_HOLDINGS": str(daily_flagged),
        }
    )

    before = set(_latest_run_ids(200))

    print(f"[bootstrap] stages 1-5 | top_n={bootstrap_top_n} | model={model}")
    _run([sys.executable, "run.py", "--mode", "pipeline", "--stages", "1-5"], env=env)

    # daily-lite runs should not spend on stage4/5
    print(f"[daily-lite] 5 runs | new={daily_new} flagged={daily_flagged} | model={model}")
    for i in range(5):
        print(f"  day {i+1}/5")
        _run([sys.executable, "run.py", "--mode", "daily"], env=env)

    after = set(_latest_run_ids(400))
    new_ids = [rid for rid in sorted(after - before) if (RUNS_DIR / rid / "run_summary.json").exists()]
    if not new_ids:
        raise SystemExit("No new run summaries found.")

    totals: list[RunTotals] = []
    for rid in new_ids:
        s = _load_summary(rid)
        in_t, out_t = _token_totals_from_summary(s)
        totals.append(RunTotals(rid, in_t, out_t))

    grand_in = sum(t.input_tokens for t in totals)
    grand_out = sum(t.output_tokens for t in totals)

    print("\nRun token totals:")
    for t in totals:
        print(f"  {t.run_id}: in={t.input_tokens} out={t.output_tokens} total={t.input_tokens+t.output_tokens}")
    print(f"\nGRAND TOTAL: in={grand_in} out={grand_out} total={grand_in+grand_out}")


if __name__ == "__main__":
    main()
