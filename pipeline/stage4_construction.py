"""
Stage 4: Portfolio Construction — Deterministic Optimizer + LLM Rationale

Key changes from original:
1. Weight allocation is DETERMINISTIC: weight = normalized(EV * conviction)
2. Hard constraints (sector caps, position bounds) are mathematically enforced
3. LLM only generates rationale text (position reasoning, exit triggers)
4. Validation checks throw errors if constraints violated
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


import config
from agents.portfolio_agent import PortfolioAgent
from agents.debate_scorer import compute_conviction
from data.fetcher import get_macro_context
from pipeline.validators import validate_sector_caps, validate_position_bounds, validate_allocation_sum
from llm.providers import parse_model_provider

logger = logging.getLogger(__name__)


import math

# Unit convention for this module: `allocation_pct` is stored in percent (0-100),
# not decimal (0-1). Config thresholds (MIN_POSITION_PCT etc.) are decimal and
# get multiplied by 100 at use sites. Do not mix.

def _compute_deterministic_weights(candidates: list[dict]) -> list[dict]:
    """
    Compute portfolio weights deterministically using exponential scaling (softmax-like).

    Provides strict separation across narrow EV distributions.
    Penalizes very low EV aggressively (threshold and factor from config).
    """
    temperature = config.STAGE4_WEIGHT_TEMPERATURE
    ev_penalty_threshold = config.STAGE4_EV_PENALTY_THRESHOLD
    ev_penalty_factor = config.STAGE4_EV_PENALTY_FACTOR

    for stock in candidates:
        ev = float(stock.get("probability_weighted_return_12m") or 0)
        conviction = float(stock.get("conviction") or 0.5)

        penalty = ev_penalty_factor if ev < ev_penalty_threshold else 1.0

        base_score = ev * conviction * penalty

        clamped_score = max(-10.0, min(10.0, base_score / temperature))
        stock["_raw_weight"] = math.exp(clamped_score)

    total_weight = sum(s["_raw_weight"] for s in candidates)
    if total_weight <= 0:
        total_weight = 1.0

    for stock in candidates:
        stock["allocation_pct"] = (stock["_raw_weight"] / total_weight) * 100.0

    return candidates


def _enforce_position_bounds(positions: list[dict]) -> list[dict]:
    """Enforce min/max position size constraints."""
    min_pct = config.MIN_POSITION_PCT * 100
    max_pct = config.MAX_POSITION_PCT * 100

    for p in positions:
        alloc = float(p.get("allocation_pct", 0))
        alloc = max(min_pct, min(max_pct, alloc))
        p["allocation_pct"] = alloc

    return positions


def _enforce_sector_caps(positions: list[dict]) -> list[dict]:
    """Enforce sector allocation caps with iterative rebalancing."""
    max_sector_pct_val = config.MAX_SECTOR_PCT * 100
    min_pct_val = config.MIN_POSITION_PCT * 100
    max_pos_pct_val = config.MAX_POSITION_PCT * 100

    for iteration in range(30):  # Enough iterations for convergence
        # Normalize to 100% at the start of each iteration
        total_alloc = sum(p.get("allocation_pct", 0) for p in positions) or 100.0
        if abs(total_alloc - 100.0) > 0.5:
            for p in positions:
                p["allocation_pct"] = p.get("allocation_pct", 0) / total_alloc * 100

        # Compute sector totals
        sector_totals: dict[str, float] = {}
        for p in positions:
            sector = p.get("sector", "Unknown")
            sector_totals[sector] = sector_totals.get(sector, 0) + p.get("allocation_pct", 0)

        # Find overweight sectors
        overweight_sectors = [
            (sector, pct) for sector, pct in sector_totals.items()
            if pct > max_sector_pct_val + 0.01
        ]
        if not overweight_sectors:
            break

        # Process the most overweight sector first
        overweight_sectors.sort(key=lambda x: x[1], reverse=True)
        sector, pct = overweight_sectors[0]
        excess = pct - max_sector_pct_val

        # Cut from largest positions in the overweight sector
        sector_positions = [p for p in positions if p.get("sector", "Unknown") == sector]
        sector_positions.sort(key=lambda p: p.get("allocation_pct", 0), reverse=True)

        total_cut = 0.0
        for p in sector_positions:
            reducible = max(0.0, p.get("allocation_pct", 0) - min_pct_val)
            cut = min(reducible, excess - total_cut)
            if cut > 0.01:
                p["allocation_pct"] -= cut
                total_cut += cut
            if total_cut >= excess - 0.01:
                break

        if total_cut < 0.01:
            # Can't reduce further (all at minimum) — break to avoid infinite loop
            break

        # Redistribute cut amount across non-overweight sectors
        sector_totals_new: dict[str, float] = {}
        for p in positions:
            s = p.get("sector", "Unknown")
            sector_totals_new[s] = sector_totals_new.get(s, 0) + p.get("allocation_pct", 0)

        redistributable = total_cut
        others = [
            p for p in positions
            if p.get("sector", "Unknown") != sector
        ]
        others.sort(key=lambda p: p.get("ev_12m_return", 0), reverse=True)

        for p in others:
            if redistributable <= 0.01:
                break
            p_sector = p.get("sector", "Unknown")
            sector_current = sector_totals_new.get(p_sector, 0)
            sector_room = max(0.0, max_sector_pct_val - sector_current)
            pos_room = max(0.0, max_pos_pct_val - p.get("allocation_pct", 0))
            add = min(redistributable, sector_room, pos_room)
            if add > 0.01:
                p["allocation_pct"] += add
                sector_totals_new[p_sector] = sector_totals_new.get(p_sector, 0) + add
                redistributable -= add

    return positions


def _normalize_to_100(positions: list[dict]) -> list[dict]:
    """Normalize allocations to sum to exactly 100%."""
    total = sum(p.get("allocation_pct", 0) for p in positions)
    if total <= 0:
        equal = 100.0 / len(positions) if positions else 0
        for p in positions:
            p["allocation_pct"] = equal
        return positions

    if abs(total - 100.0) > 0.5:
        for p in positions:
            p["allocation_pct"] = (p.get("allocation_pct", 0) / total) * 100

    return positions


def _solve_constraints(positions: list[dict], max_outer_iters: int = 10) -> list[dict]:
    """
    Run position-bound + sector-cap + sum-to-100 enforcement to joint convergence.
    Raises RuntimeError if constraints cannot be jointly satisfied.
    """
    if not positions:
        return positions

    max_sector = config.MAX_SECTOR_PCT * 100
    min_pos = config.MIN_POSITION_PCT * 100
    max_pos = config.MAX_POSITION_PCT * 100

    for _ in range(max_outer_iters):
        positions = _enforce_position_bounds(positions)
        positions = _enforce_sector_caps(positions)
        positions = _normalize_to_100(positions)

        # Check joint satisfaction
        total = sum(p.get("allocation_pct", 0) for p in positions)
        if abs(total - 100.0) > 0.5:
            continue
        if any(p.get("allocation_pct", 0) < min_pos - 0.01 or p.get("allocation_pct", 0) > max_pos + 0.01 for p in positions):
            continue
        sector_tot: dict[str, float] = {}
        for p in positions:
            s = p.get("sector", "Unknown")
            sector_tot[s] = sector_tot.get(s, 0) + p.get("allocation_pct", 0)
        if any(v > max_sector + 0.01 for v in sector_tot.values()):
            continue
        return positions

    raise RuntimeError(
        f"Stage 4 constraint solver failed to converge after {max_outer_iters} iterations. "
        f"Inputs may be infeasible (e.g., too few sectors for MAX_SECTOR_PCT)."
    )


class Stage4PortfolioConstruction:
    """
    Stage 4: Deterministic portfolio optimizer.
    Selects positions and computes weights mathematically.
    LLM generates rationale only.
    """

    def __init__(self):
        self.agent = PortfolioAgent()

    def run(
        self,
        scenario_models: list[dict],
        current_portfolio: dict = None,
        save_path: Optional[Path] = None,
    ) -> dict:
        """
        Construct the portfolio using deterministic optimization.

        Args:
            scenario_models: Stage 3 output (with debate scores)
            current_portfolio: Existing portfolio (for context)
            save_path: Where to save results

        Returns:
            Portfolio construction dict
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("STAGE 4: Deterministic Portfolio Construction")
        logger.info(f"         {len(scenario_models)} candidates → {config.MAX_POSITIONS} positions")
        logger.info("=" * 60)

        # Filter: only stocks that made it through selection pressure
        eligible = [
            m for m in scenario_models
            if m.get("probability_weighted_return_12m") is not None
            and m.get("probability_weighted_return_12m") > config.MIN_EXPECTED_RETURN
            and not m.get("error")
            and str(m.get("symbol") or "").strip()
            and m.get("_portfolio_eligible", True)
        ]

        # Sort by EV descending
        eligible.sort(key=lambda x: x.get("probability_weighted_return_12m", 0), reverse=True)

        logger.info(f"Eligible candidates (positive EV + portfolio eligible): {len(eligible)} / {len(scenario_models)}")

        if len(eligible) < config.MIN_POSITIONS:
            logger.warning(
                f"Only {len(eligible)} eligible stocks. "
                f"Relaxing portfolio eligibility filter."
            )
            # Fallback: include all positive-EV stocks
            eligible = [
                m for m in scenario_models
                if m.get("probability_weighted_return_12m") is not None
                and m.get("probability_weighted_return_12m") > config.MIN_EXPECTED_RETURN
                and not m.get("error")
                and str(m.get("symbol") or "").strip()
            ]
            eligible.sort(key=lambda x: x.get("probability_weighted_return_12m", 0), reverse=True)

        # Select top N positions
        selected = eligible[:config.MAX_POSITIONS]

        if len(selected) < config.MIN_POSITIONS:
            raise RuntimeError(
                f"Stage 4 cannot construct portfolio: only {len(selected)} eligible candidates "
                f"after fallback (need >= {config.MIN_POSITIONS}). "
                f"Inputs={len(scenario_models)}, eligible={len(eligible)}."
            )

        # ── Deterministic weight computation + joint constraint solver ──
        selected = _compute_deterministic_weights(selected)
        selected = _solve_constraints(selected)

        # Build portfolio positions
        portfolio_value = config.PAPER_PORTFOLIO_VALUE_INR
        positions = []
        for i, stock in enumerate(selected):
            alloc_pct = round(stock.get("allocation_pct", 0), 2)
            positions.append({
                "rank": i + 1,
                "symbol": str(stock.get("symbol") or "").strip().upper(),
                "company_name": str(stock.get("company_name") or "").strip(),
                "sector": stock.get("sector", "Unknown"),
                "allocation_pct": alloc_pct,
                "allocation_inr": round(alloc_pct / 100 * portfolio_value),
                "current_price": stock.get("current_price") or stock.get("price"),
                "ev_12m_return": stock.get("probability_weighted_return_12m", 0),
                "conviction": "High" if stock.get("conviction", 0) > 0.7 else ("Medium" if stock.get("conviction", 0) > 0.4 else "Low"),
                "conviction_score": stock.get("conviction", 0),
                "debate_scores": stock.get("debate_scores", {}),
                "position_rationale": stock.get("investment_thesis") or "Selected by EV×conviction ranking.",
                "entry_note": "",
                "exit_trigger": stock.get("thesis_invalidation", "Thesis breach"),
            })

        # ── Generate rationale text via LLM (optional, non-blocking) ──
        if not config.STAGE4_DETERMINISTIC_OPTIMIZER:
            try:
                logger.info("Generating position rationale via LLM...")
                macro = get_macro_context()
                llm_output = self.agent.construct(
                    scenario_models=eligible,
                    current_portfolio=current_portfolio,
                    macro_context=macro,
                )
                # Extract rationale from LLM output and merge
                llm_positions = {
                    p.get("symbol", "").upper(): p
                    for p in llm_output.get("portfolio", [])
                }
                for pos in positions:
                    llm_pos = llm_positions.get(pos["symbol"], {})
                    if llm_pos.get("position_rationale"):
                        pos["position_rationale"] = llm_pos["position_rationale"]
                    if llm_pos.get("entry_note"):
                        pos["entry_note"] = llm_pos["entry_note"]
                    if llm_pos.get("exit_trigger"):
                        pos["exit_trigger"] = llm_pos["exit_trigger"]
            except Exception as e:
                logger.warning(f"LLM rationale generation failed, using defaults: {e}")

        # ── Build excluded stocks list ──
        excluded = []
        selected_symbols = {p["symbol"] for p in positions}
        for stock in eligible:
            sym = str(stock.get("symbol") or "").strip().upper()
            if sym and sym not in selected_symbols:
                excluded.append({
                    "symbol": sym,
                    "ev_12m_return": stock.get("probability_weighted_return_12m"),
                    "reason": "Below top-N by EV×conviction. " +
                              (f"Conviction too low ({stock.get('conviction', 0):.2f})."
                               if stock.get("conviction", 0) < 0.4
                               else "Outranked by higher EV stocks."),
                })

        # ── Compute portfolio summary ──
        sector_breakdown = {}
        for p in positions:
            sector = p.get("sector", "Unknown")
            sector_breakdown[sector] = round(
                sector_breakdown.get(sector, 0) + p.get("allocation_pct", 0), 1
            )

        weighted_ev = sum(
            p.get("allocation_pct", 0) / 100 * p.get("ev_12m_return", 0)
            for p in positions
        )

        portfolio_summary = {
            "total_positions": len(positions),
            "total_allocation_pct": round(sum(p.get("allocation_pct", 0) for p in positions), 1),
            "weighted_avg_ev_return": round(weighted_ev, 4),
            "vs_nifty50_alpha": round(weighted_ev - config.BENCHMARK_ANNUAL_RETURN_TARGET, 4),
            "sector_breakdown": sector_breakdown,
            "construction_method": "deterministic_ev_conviction",
        }

        portfolio_data = {
            "portfolio": positions,
            "portfolio_summary": portfolio_summary,
            "excluded_stocks": excluded[:10],
            "construction_notes": (
                f"Deterministic construction: weight = EV × conviction. "
                f"Received {len(scenario_models)} candidates, {len(eligible)} eligible, "
                f"selected top {len(positions)}. Hard constraints enforced: "
                f"sector≤{config.MAX_SECTOR_PCT*100:.0f}%, "
                f"position [{config.MIN_POSITION_PCT*100:.0f}%, {config.MAX_POSITION_PCT*100:.0f}%]."
            ),
        }

        # ── Validate ──
        errors = []
        errors.extend(validate_sector_caps(positions, config.MAX_SECTOR_PCT))
        errors.extend(validate_position_bounds(positions, config.MIN_POSITION_PCT, config.MAX_POSITION_PCT))
        errors.extend(validate_allocation_sum(positions))

        if errors:
            logger.warning(f"Post-construction validation found {len(errors)} issue(s): {errors}")
            positions = _solve_constraints(positions)
            for p in positions:
                p["allocation_pct"] = round(p["allocation_pct"], 2)
                p["allocation_inr"] = round(p["allocation_pct"] / 100 * portfolio_value)
            portfolio_data["portfolio"] = positions
            re_errors = []
            re_errors.extend(validate_sector_caps(positions, config.MAX_SECTOR_PCT))
            re_errors.extend(validate_position_bounds(positions, config.MIN_POSITION_PCT, config.MAX_POSITION_PCT))
            re_errors.extend(validate_allocation_sum(positions))
            if re_errors:
                raise RuntimeError(f"Stage 4 constraint violation after re-solve: {re_errors}")

        self._validate_candidate_identity(positions)

        duration = (datetime.now() - start_time).total_seconds()

        portfolio_data["stage"] = 4
        portfolio_data["run_time"] = datetime.now().isoformat()
        portfolio_data["duration_seconds"] = duration
        portfolio_data["token_usage"] = self.agent.get_token_usage()
        portfolio_data["eligible_candidates"] = len(eligible)

        if save_path:
            with open(save_path, "w") as f:
                json.dump(portfolio_data, f, indent=2, default=str)
            logger.info(f"Stage 4 results saved to {save_path}")

        logger.info(f"\n✅ Stage 4 complete in {duration:.0f}s")
        logger.info(f"   Positions selected: {len(positions)}")
        logger.info(f"   Portfolio EV: {weighted_ev*100:+.1f}%")
        logger.info(f"   Token usage: {self.agent.get_token_usage()}")

        if positions:
            logger.info(f"\n   Selected Portfolio:")
            logger.info(f"   {'#':>3} {'Symbol':12} {'Alloc':>6} {'EV':>7}  {'Conv':>5}  Sector")
            logger.info(f"   {'-'*60}")
            for p in positions:
                logger.info(
                    f"   {p.get('rank', '?'):>3} {p.get('symbol', '?'):12} "
                    f"{p.get('allocation_pct', 0):>5.1f}%  "
                    f"{p.get('ev_12m_return', 0)*100:>+6.1f}%  "
                    f"{p.get('conviction', 'M'):>5}  "
                    f"{p.get('sector', 'Unknown')}"
                )

        return portfolio_data

    def _validate_candidate_identity(self, positions: list[dict]) -> None:
        missing = []
        for idx, stock in enumerate(positions):
            symbol = str(stock.get("symbol") or "").strip().upper()
            company_name = str(stock.get("company_name") or "").strip()
            if not symbol or not company_name:
                missing.append((idx, symbol, company_name))
        if missing:
            sample = ", ".join([f"{idx}:{sym or '?'}:{name or '?'}" for idx, sym, name in missing[:5]])
            raise ValueError(f"Stage 4 produced positions with missing identity: {sample}")
