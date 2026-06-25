"""
Master Orchestrator — runs the full 5-stage pipeline end to end.
Chains: Stage 1 → 2 → 3 → 4 → 5
Saves all intermediate outputs and updates portfolio.json + trades_log.json.
"""

from pathlib import Path
import json
import logging
import time
from datetime import datetime


import config
from pipeline.stage1_screening import Stage1Screener
from pipeline.stage2_adversarial import Stage2AdversarialResearch
from pipeline.stage3_scenarios import Stage3ScenarioModeling
from pipeline.stage4_construction import Stage4PortfolioConstruction
from pipeline.stage5_rebalance import Stage5Rebalancer
from pipeline.validators import validate_pipeline_output, PipelineValidationError
from persistence import db as pdb
from llm.providers import parse_model_provider

logger = logging.getLogger(__name__)


def _validate_stage5_before_persist(rebalance_data: dict) -> None:
    """
    Gate before any portfolio/trade persistence.
    Raises RuntimeError if Stage 5 output is structurally unsafe.
    Empty or error output must NEVER overwrite existing portfolio.
    """
    if not isinstance(rebalance_data, dict):
        raise RuntimeError(f"Stage 5 output is not a dict (got {type(rebalance_data)})")
    if "error" in rebalance_data and not rebalance_data.get("trades"):
        raise RuntimeError(f"Stage 5 parse failed — not persisting: {rebalance_data.get('error', '')[:200]}")
    trades = rebalance_data.get("trades")
    if trades is None:
        raise RuntimeError("Stage 5 output missing 'trades' key — refusing to persist")
    if not isinstance(trades, list):
        raise RuntimeError(f"Stage 5 'trades' is not a list — refusing to persist")
    for i, t in enumerate(trades):
        if not isinstance(t, dict):
            raise RuntimeError(f"Stage 5 trade[{i}] is not a dict")
        if not t.get("symbol"):
            raise RuntimeError(f"Stage 5 trade[{i}] missing 'symbol'")
        if (t.get("action") or "").upper() not in ("BUY", "SELL", "HOLD"):
            raise RuntimeError(f"Stage 5 trade[{i}] invalid action: {t.get('action')!r}")


def _load_current_portfolio() -> dict:
    """Load the current portfolio from disk."""
    try:
        portfolio = pdb.load_current_portfolio()
        if portfolio.get("positions"):
            return portfolio
    except Exception:
        pass
    if config.PORTFOLIO_FILE.exists():
        with open(config.PORTFOLIO_FILE) as f:
            return json.load(f)
    return {}


def _save_portfolio(portfolio_data: dict, rebalance_data: dict, run_id: str) -> None:
    """Persist the new portfolio state to disk."""
    # Build positions list from rebalance trades
    trades = rebalance_data.get("trades", [])
    new_positions_map = {p["symbol"]: p for p in portfolio_data.get("portfolio", [])}

    positions = []
    for trade in trades:
        action = trade.get("action")
        symbol = trade.get("symbol")

        if action in ("BUY", "HOLD"):
            pos = new_positions_map.get(symbol, trade)
            positions.append({
                "symbol": symbol,
                "company_name": trade.get("company_name") or pos.get("company_name"),
                "sector": trade.get("sector") or pos.get("sector"),
                "allocation_pct": trade.get("target_allocation_pct") or trade.get("new_target_allocation_pct") or pos.get("allocation_pct"),
                "allocation_inr": trade.get("target_allocation_inr") or pos.get("allocation_inr"),
                "current_price": trade.get("current_price") or pos.get("current_price"),
                "entry_price": trade.get("current_price") if action == "BUY" else trade.get("entry_price"),
                "ev_12m_return": trade.get("ev_12m_return") or pos.get("ev_12m_return"),
                "conviction": trade.get("conviction") or pos.get("conviction", "Medium"),
                "rationale": trade.get("rationale") or pos.get("position_rationale"),
                "entry_note": trade.get("entry_note", ""),
                "exit_trigger": trade.get("exit_trigger") or pos.get("exit_trigger"),
                "action": action,
                "last_updated": datetime.now().isoformat(),
            })

    portfolio = {
        "last_run": datetime.now().isoformat(),
        "portfolio_value_inr": config.PAPER_PORTFOLIO_VALUE_INR,
        "positions": positions,
        "summary": portfolio_data.get("portfolio_summary", {}),
    }

    # Commit DB first; JSON is only written if DB persistence succeeds, so
    # dashboard (DB reader) and JSON file stay in sync.
    try:
        pdb.replace_current_positions(positions)
        pdb.save_positions(run_id, positions)
    except Exception as e:
        logger.error("DB write failed for portfolio: %s — NOT writing JSON to keep state consistent", e)
        raise

    tmp_path = config.PORTFOLIO_FILE.with_suffix(config.PORTFOLIO_FILE.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)
    tmp_path.replace(config.PORTFOLIO_FILE)
    logger.info(f"Portfolio saved to {config.PORTFOLIO_FILE}")


def _append_trades_log(rebalance_data: dict, run_id: str) -> None:
    """Append this cycle's trades to the persistent trades log."""
    existing = []
    if config.TRADES_LOG_FILE.exists():
        with open(config.TRADES_LOG_FILE) as f:
            existing = json.load(f)

    entry = {
        "run_id": run_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "trades": rebalance_data.get("trades", []),
        "summary": rebalance_data.get("rebalance_summary", {}),
        "notes": rebalance_data.get("rebalance_notes", ""),
    }
    existing.append(entry)

    try:
        pdb.append_trades(run_id, rebalance_data.get("trades", []))
    except Exception as e:
        logger.error("DB write failed for trades: %s — NOT appending to JSON log", e)
        raise

    tmp_path = config.TRADES_LOG_FILE.with_suffix(config.TRADES_LOG_FILE.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    tmp_path.replace(config.TRADES_LOG_FILE)
    logger.info(f"Trades log updated ({len(existing)} total runs)")


def _append_decision_artifacts(run_id: str, stage: int, artifacts: list[dict]) -> None:
    if not artifacts:
        return
    try:
        pdb.append_decision_artifacts(run_id, stage, artifacts)
    except Exception as e:
        logger.warning("DB write failed for decision artifacts stage=%s: %s", stage, e)


def run_pipeline(
    stage_start: int = 1,
    stage_end: int = 5,
    run_id: str = None,
    mock_stage1: list = None,
    mock_stage2: dict = None,
    daily_lite: bool = False,
    daily_new_k: int | None = None,
    daily_flagged_k: int | None = None,
) -> dict:
    """
    Run the full 5-stage pipeline.

    Args:
        stage_start: Which stage to start from (1–5). Useful for resuming.
        stage_end: Which stage to end at.
        run_id: Unique run identifier (auto-generated if None).
        mock_stage1: If provided, skip Stage 1 and use this as top50.
        mock_stage2: If provided, skip Stage 2 and use this as research output.

    Returns:
        Summary dict with all stage outputs and token costs.
    """
    # Only require Anthropic key if any stage in this run uses an Anthropic model.
    stage_models = {
        1: config.STAGE1_MODEL,
        2: config.STAGE2_MODEL,
        3: config.STAGE3_MODEL,
        4: config.STAGE4_MODEL,
        5: config.STAGE5_MODEL,
    }
    needs_anthropic = False
    for s in range(stage_start, stage_end + 1):
        provider, _ = parse_model_provider(stage_models.get(s, ""))
        if provider == "anthropic":
            needs_anthropic = True
            break
    if needs_anthropic and not config.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set, but this run uses an Anthropic model.\n"
            "Either set ANTHROPIC_API_KEY in .env or switch models to gemini:/groq:/openai: for a no-tools test run.\n"
            "For Lightning CloudSpaces, set OPENAI_COMPAT_BASE_URL and use openai:<model> or the raw hosted model name.\n"
            "See .env.example for reference."
        )

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.PIPELINE_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_start = datetime.now()

    # Recover stale runs (stuck in "running" > 2h) before starting a new one
    try:
        pdb.mark_stale_runs(cutoff_hours=2)
    except Exception as e:
        logger.warning("Could not recover stale runs: %s", e)

    try:
        pdb.upsert_run(
            run_id,
            started_at=pipeline_start.isoformat(),
            stage_start=stage_start,
            stage_end=stage_end,
            status="running",
            meta={"run_dir": str(run_dir)},
        )
    except Exception as e:
        logger.error("DB: failed to record run start for %s: %s — continuing", run_id, e)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║        INDIAN AI PORTFOLIO — PIPELINE RUN                ║")
    logger.info(f"║  Run ID: {run_id:<49}║")
    logger.info(f"║  Time:   {pipeline_start.strftime('%Y-%m-%d %H:%M:%S'):<49}║")
    logger.info(f"║  Budget: ₹{config.PAPER_PORTFOLIO_VALUE_INR:>10,.0f}                                  ║")
    logger.info("╚══════════════════════════════════════════════════════════╝\n")

    results = {"run_id": run_id, "start_time": pipeline_start.isoformat()}
    token_totals = {"input": 0, "output": 0, "cost_usd": 0.0}

    def _accum_usage(usage: dict | None) -> None:
        if not isinstance(usage, dict):
            return
        token_totals["input"] += int(usage.get("input_tokens") or 0)
        token_totals["output"] += int(usage.get("output_tokens") or 0)
        token_totals["cost_usd"] += float(usage.get("estimated_cost_usd") or 0.0)

    try:
        # ── Stage 1: Screening ───────────────────────────────────────────────
        top50 = mock_stage1
        if stage_start <= 1 <= stage_end and top50 is None:
            s1 = Stage1Screener()
            top50 = s1.run(save_path=run_dir / "stage1_output.json")
            usage = s1.agent.get_token_usage()
            _accum_usage(usage)
            results["stage1"] = {"top50_count": len(top50), "token_usage": usage}
            _append_decision_artifacts(
                run_id,
                1,
                [
                    {
                        "kind": "stage1_selection",
                        "symbol": row.get("symbol"),
                        "payload": {
                            "rank": row.get("rank"),
                            "symbol": row.get("symbol"),
                            "company_name": row.get("company_name"),
                            "sector": row.get("sector"),
                            "advance_reason": row.get("advance_reason"),
                            "flags": row.get("flags", []),
                            "composite_score": row.get("composite_score"),
                            "quant_rank": row.get("quant_rank"),
                            "analyst_rank": row.get("analyst_rank"),
                        },
                    }
                    for row in top50
                ],
            )
            logger.info(f"\n{'─'*60}")
        elif top50:
            logger.info("Stage 1: Using provided mock data (skipped)")

        # Daily-lite selection: still run Stage 1 (full sweep) but only send a tiny shortlist to Stage 2/3.
        if daily_lite and top50:
            daily_new_k = config.DAILY_LITE_NEW_CANDIDATES if daily_new_k is None else int(daily_new_k)
            daily_flagged_k = config.DAILY_LITE_FLAGGED_HOLDINGS if daily_flagged_k is None else int(daily_flagged_k)
            daily_new_k = max(0, daily_new_k)
            daily_flagged_k = max(0, daily_flagged_k)

            try:
                current = pdb.load_current_portfolio()
                held = {(p.get("symbol") or "").strip().upper() for p in current.get("positions", [])}
            except Exception as e:
                raise RuntimeError(f"Failed to load current portfolio from DB — cannot compute daily-lite diff safely: {e}") from e

            flagged: list[str] = []
            try:
                flagged = pdb.get_recent_flagged_symbols(max_symbols=daily_flagged_k, lookback_hours=48)
            except Exception as e:
                logger.error("Failed to load flagged symbols from DB: %s — proceeding with empty list", e)
                flagged = []

            # Pick top "new" names from Stage 1 top50 that are not currently held and not already flagged.
            new: list[str] = []
            for row in top50:
                sym = (row.get("symbol") or "").strip().upper()
                if not sym:
                    continue
                if sym in held:
                    continue
                if sym in flagged:
                    continue
                new.append(sym)
                if len(new) >= daily_new_k:
                    break

            shortlist = flagged + new
            # Build a Stage-2 input list (dicts). If a flagged symbol isn't in top50, create a minimal record.
            by_symbol = {(r.get("symbol") or "").strip().upper(): r for r in top50 if (r.get("symbol") or "").strip()}
            stage2_list: list[dict] = []
            for sym in shortlist:
                rec = by_symbol.get(sym)
                if rec:
                    stage2_list.append(rec)
                    continue
                # Minimal fallback record; Stage2 no-tools mode fetches fundamentals/news itself anyway.
                stage2_list.append({"symbol": sym, "company_name": sym, "sector": "Unknown"})

            results["daily_lite"] = {
                "flagged_symbols": flagged,
                "new_symbols": new,
                "shortlist": shortlist,
            }
            top50 = stage2_list

        if stage_end < 2:
            results["end_stage"] = 1
            return _finalize(results, token_totals, pipeline_start, run_dir, stage_start, stage_end)

        # ── Stage 2: Adversarial Research ───────────────────────────────────
        stage2_research = mock_stage2
        if stage2_research:
            logger.info("Stage 2: Using provided research data (skipped)")
        elif stage_start <= 2 <= stage_end and top50:
            s2_workers = config.DAILY_LITE_S2_MAX_WORKERS if daily_lite else config.PIPELINE_S2_MAX_WORKERS
            s2 = Stage2AdversarialResearch(max_workers=s2_workers)
            stage2_research = s2.run(
                top50,
                n_stocks=(len(top50) if daily_lite else None),
                run_id=run_id,
                save_path=run_dir / "stage2_output.json",
            )
            # Aggregate token usage from all debate agents.
            s2_usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
            for r in stage2_research.values():
                if not isinstance(r, dict):
                    continue
                u = r.get("token_usage") or {}
                s2_usage_total["input_tokens"] += int(u.get("input_tokens") or 0)
                s2_usage_total["output_tokens"] += int(u.get("output_tokens") or 0)
                s2_usage_total["estimated_cost_usd"] += float(u.get("estimated_cost_usd") or 0.0)
            s2_usage_total["total_tokens"] = s2_usage_total["input_tokens"] + s2_usage_total["output_tokens"]
            s2_usage_total["estimated_cost_usd"] = round(s2_usage_total["estimated_cost_usd"], 4)
            _accum_usage(s2_usage_total)

            # Count debate statistics
            debate_stats = {"stance_changes": 0, "debates_with_disagreement": 0}
            for r in stage2_research.values():
                if not isinstance(r, dict):
                    continue
                transcript = r.get("debate_transcript", {})
                debate_stats["stance_changes"] += transcript.get("stance_changes_count", 0)
                scores = r.get("debate_scores", {})
                dist = scores.get("stance_distribution", {})
                if dist.get("BUY", 0) > 0 and dist.get("SELL", 0) > 0:
                    debate_stats["debates_with_disagreement"] += 1

            results["stage2"] = {
                "stocks_researched": len(stage2_research),
                "debate_system": {
                    "agents": config.DEBATE_AGENTS,
                    "rounds": config.DEBATE_ROUNDS,
                },
                "debate_stats": debate_stats,
                "token_usage": s2_usage_total,
            }
            stage2_artifacts = []
            for sym, record in (stage2_research or {}).items():
                stage2_artifacts.append(
                    {
                        "kind": "stage2_debate_summary",
                        "symbol": sym,
                        "payload": {
                            "symbol": sym,
                            "company_name": record.get("company_name"),
                            "sector": record.get("sector"),
                            "bull_thesis": record.get("bull_thesis"),
                            "bear_thesis": record.get("bear_thesis"),
                            "debate_scores": record.get("debate_scores", {}),
                            "conviction": record.get("conviction"),
                            "synthesis_packet": record.get("synthesis_packet"),
                        },
                    }
                )
            _append_decision_artifacts(run_id, 2, stage2_artifacts)
            logger.info(f"\n{'─'*60}")

        if stage_end < 3:
            results["end_stage"] = 2
            return _finalize(results, token_totals, pipeline_start, run_dir, stage_start, stage_end)

        # ── Stage 3: Scenario Modeling ───────────────────────────────────────
        scenario_models = None
        if stage_start <= 3 <= stage_end and stage2_research:
            s3_workers = config.DAILY_LITE_S3_MAX_WORKERS if daily_lite else config.PIPELINE_S3_MAX_WORKERS
            s3 = Stage3ScenarioModeling(max_workers=s3_workers)
            scenario_models = s3.run(
                stage2_research,
                top50 or [],
                save_path=run_dir / "stage3_output.json",
            )
            s3_usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
            for m in scenario_models or []:
                u = (m or {}).get("_token_usage") or {}
                s3_usage_total["input_tokens"] += int(u.get("input_tokens") or 0)
                s3_usage_total["output_tokens"] += int(u.get("output_tokens") or 0)
                s3_usage_total["estimated_cost_usd"] += float(u.get("estimated_cost_usd") or 0.0)
            s3_usage_total["total_tokens"] = s3_usage_total["input_tokens"] + s3_usage_total["output_tokens"]
            s3_usage_total["estimated_cost_usd"] = round(s3_usage_total["estimated_cost_usd"], 4)
            _accum_usage(s3_usage_total)
            results["stage3"] = {
                "modeled_count": len(scenario_models),
                "positive_ev_count": len([m for m in scenario_models if (m.get("probability_weighted_return_12m") or 0) > 0]),
                "token_usage": s3_usage_total,
            }
            _append_decision_artifacts(
                run_id,
                3,
                [
                    {
                        "kind": "stage3_scenario_model",
                        "symbol": model.get("symbol"),
                        "payload": {
                            "symbol": model.get("symbol"),
                            "company_name": model.get("company_name"),
                            "sector": model.get("sector"),
                            "probability_weighted_return_12m": model.get("probability_weighted_return_12m"),
                            "net_return_after_costs": model.get("net_return_after_costs"),
                            "recommendation": model.get("recommendation"),
                            "investment_thesis": model.get("investment_thesis"),
                            "key_catalysts_next_90_days": model.get("key_catalysts_next_90_days", []),
                            "key_risks": model.get("key_risks", []),
                            "thesis_invalidation": model.get("thesis_invalidation"),
                            "analyst_consensus": model.get("analyst_consensus", {}),
                            "scenarios": model.get("scenarios", {}),
                        },
                    }
                    for model in (scenario_models or [])
                ],
            )
            logger.info(f"\n{'─'*60}")

        if stage_end < 4:
            results["end_stage"] = 3
            return _finalize(results, token_totals, pipeline_start, run_dir, stage_start, stage_end)

        # ── Stage 4: Portfolio Construction ──────────────────────────────────
        portfolio_construction = None
        current_portfolio = _load_current_portfolio()

        if stage_start <= 4 <= stage_end and scenario_models:
            s4 = Stage4PortfolioConstruction()
            portfolio_construction = s4.run(
                scenario_models,
                current_portfolio=current_portfolio,
                save_path=run_dir / "stage4_output.json",
            )
            usage = s4.agent.get_token_usage()
            _accum_usage(usage)
            results["stage4"] = {
                "positions_selected": len(portfolio_construction.get("portfolio", [])),
                "portfolio_ev": portfolio_construction.get("portfolio_summary", {}).get("weighted_avg_ev_return"),
                "token_usage": usage,
            }
            stage4_artifacts = []
            for pos in portfolio_construction.get("portfolio", []):
                stage4_artifacts.append(
                    {
                        "kind": "stage4_position",
                        "symbol": pos.get("symbol"),
                        "payload": {
                            "rank": pos.get("rank"),
                            "symbol": pos.get("symbol"),
                            "company_name": pos.get("company_name"),
                            "sector": pos.get("sector"),
                            "allocation_pct": pos.get("allocation_pct"),
                            "allocation_inr": pos.get("allocation_inr"),
                            "ev_12m_return": pos.get("ev_12m_return"),
                            "conviction": pos.get("conviction"),
                            "position_rationale": pos.get("position_rationale"),
                            "entry_note": pos.get("entry_note"),
                            "exit_trigger": pos.get("exit_trigger"),
                        },
                    }
                )
            for ex in portfolio_construction.get("excluded_stocks", []) or []:
                stage4_artifacts.append(
                    {
                        "kind": "stage4_exclusion",
                        "symbol": ex.get("symbol"),
                        "payload": ex,
                    }
                )
            if portfolio_construction.get("construction_notes"):
                stage4_artifacts.append(
                    {
                        "kind": "stage4_construction_notes",
                        "symbol": None,
                        "payload": {
                            "construction_notes": portfolio_construction.get("construction_notes"),
                            "portfolio_summary": portfolio_construction.get("portfolio_summary", {}),
                        },
                    }
                )
            _append_decision_artifacts(run_id, 4, stage4_artifacts)
            logger.info(f"\n{'─'*60}")

            # ── Inter-stage Validation ──
            portfolio_positions = portfolio_construction.get("portfolio", [])
            debate_transcripts = [
                r.get("debate_transcript", {})
                for r in (stage2_research or {}).values()
                if isinstance(r, dict) and r.get("debate_transcript")
            ]
            strict = getattr(config, "PIPELINE_STRICT_VALIDATION", True)
            try:
                validation_errors = validate_pipeline_output(
                    portfolio=portfolio_positions,
                    scenario_results=scenario_models or [],
                    debate_results=debate_transcripts,
                    max_sector_pct=config.MAX_SECTOR_PCT,
                    min_position_pct=config.MIN_POSITION_PCT,
                    max_position_pct=config.MAX_POSITION_PCT,
                    raise_on_error=strict,
                )
                if validation_errors:
                    results["validation_warnings"] = validation_errors
                    logger.warning("Inter-stage validation found %d issue(s)", len(validation_errors))
                else:
                    logger.info("✅ Inter-stage validation passed")
            except PipelineValidationError:
                raise  # let orchestrator outer catch handle it as a failed run
            except Exception as e:
                logger.error("Inter-stage validation crashed: %s", e)
                if strict:
                    raise

        if stage_end < 5:
            results["end_stage"] = 4
            return _finalize(results, token_totals, pipeline_start, run_dir, stage_start, stage_end)

        # ── Stage 5: Rebalancing ─────────────────────────────────────────────
        if stage_start <= 5 <= stage_end and portfolio_construction:
            if not portfolio_construction.get("portfolio"):
                logger.error("Stage 5 skipped: Stage 4 produced no portfolio positions. Existing portfolio will not be overwritten.")
                results["stage5"] = {
                    "trades": 0,
                    "skipped": True,
                    "reason": "Stage 4 produced no portfolio positions",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
                }
                return _finalize(results, token_totals, pipeline_start, run_dir, stage_start, stage_end)

            s5 = Stage5Rebalancer()
            rebalance_data = s5.run(
                portfolio_construction,
                current_portfolio=current_portfolio,
                scenario_models=scenario_models,
                save_path=run_dir / "stage5_output.json",
            )
            usage = s5.agent.get_token_usage()
            _accum_usage(usage)
            results["stage5"] = {
                "trades": len(rebalance_data.get("trades", [])),
                "token_usage": usage,
            }
            stage5_artifacts = []
            for trade in rebalance_data.get("trades", []) or []:
                stage5_artifacts.append(
                    {
                        "kind": f"stage5_trade_{(trade.get('action') or '').lower()}",
                        "symbol": trade.get("symbol"),
                        "payload": trade,
                    }
                )
            if rebalance_data.get("rebalance_notes"):
                stage5_artifacts.append(
                    {
                        "kind": "stage5_rebalance_notes",
                        "symbol": None,
                        "payload": {
                            "rebalance_notes": rebalance_data.get("rebalance_notes"),
                            "rebalance_summary": rebalance_data.get("rebalance_summary", {}),
                        },
                    }
                )
            _append_decision_artifacts(run_id, 5, stage5_artifacts)

            # Guard: validate Stage 5 output before touching portfolio state
            _validate_stage5_before_persist(rebalance_data)

            # Persist portfolio and trades
            _save_portfolio(portfolio_construction, rebalance_data, run_id)
            _append_trades_log(rebalance_data, run_id)

        return _finalize(results, token_totals, pipeline_start, run_dir, stage_start, stage_end)
    except Exception as e:
        try:
            pdb.upsert_run(
                run_id,
                started_at=pipeline_start.isoformat(),
                stage_start=stage_start,
                stage_end=stage_end,
                status="failed",
                ended_at=datetime.now().isoformat(),
                meta={"error": str(e), "run_dir": str(run_dir)},
            )
        except Exception as db_err:
            logger.error("DB: failed to record run failure for %s: %s", run_id, db_err)
        raise


def _finalize(
    results: dict,
    token_totals: dict,
    start_time: datetime,
    run_dir: Path,
    stage_start: int,
    stage_end: int,
) -> dict:
    """Compute final summary and save run metadata."""
    duration = (datetime.now() - start_time).total_seconds()
    results["duration_seconds"] = duration
    results["duration_human"] = f"{duration/3600:.1f}h" if duration > 3600 else f"{duration/60:.1f}min"
    results["total_tokens"] = token_totals
    results["run_dir"] = str(run_dir)

    # Save run summary
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    try:
        pdb.upsert_run(
            results.get("run_id", ""),
            started_at=start_time.isoformat(),
            ended_at=datetime.now().isoformat(),
            stage_start=stage_start,
            stage_end=stage_end,
            status="completed",
            meta={"duration_seconds": duration, "run_dir": str(run_dir), "token_totals": token_totals},
        )
    except Exception:
        pass

    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║                 PIPELINE COMPLETE                        ║")
    logger.info(f"║  Duration: {results['duration_human']:<50}║")
    logger.info(f"║  Est. Cost: ${token_totals['cost_usd']:>7.2f} USD{' '*41}║")
    logger.info(f"║  Output:   {str(run_dir):<50}║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    return results


def run_partial_pipeline(
    symbols: list[str],
    run_id: str = None,
) -> dict:
    """
    Partial pipeline rerun for a subset of symbols (thesis-break path).

    Runs Stage 2 → 3 → 4 on the given symbols only, skipping Stage 1 (full universe).
    Stage 5 (rebalance) is intentionally skipped — caller should review output and
    trigger a full rebalance manually if needed.

    Args:
        symbols: NSE symbols to re-research (e.g. ["RELIANCE", "INFY"])
        run_id: optional run id; auto-generated if None

    Returns:
        Partial run summary dict
    """
    if not symbols:
        logger.warning("run_partial_pipeline called with empty symbol list — no-op")
        return {"symbols": [], "status": "noop"}

    run_id = run_id or ("partial_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    logger.info("PARTIAL PIPELINE RUN: %s — symbols: %s", run_id, symbols)

    # Build minimal Stage-1-style records for the affected symbols
    mock_stage1 = [
        {"symbol": sym, "company_name": sym, "sector": "Unknown"}
        for sym in symbols
    ]

    return run_pipeline(
        stage_start=2,
        stage_end=4,
        run_id=run_id,
        mock_stage1=mock_stage1,
    )
