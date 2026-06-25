"""
Stage 3: Scenario Modeling with Debate-Derived Probabilities

Key changes from original:
1. Probabilities (P_bull, P_bear, P_base) come from debate scoring, NOT from LLM
2. LLM only generates price targets per scenario
3. Selection pressure: if >50% stocks have positive EV, normalize downward
4. Only top 30% EV stocks proceed to portfolio stage
"""

from pathlib import Path
import json
import logging
import math
import concurrent.futures
from datetime import datetime
from typing import Optional


import config
from agents.scenario_agent import ScenarioAgent
from agents.base_agent import BaseAgent
from data.fetcher import get_fundamentals
from llm.json_utils import extract_json
from llm.providers import parse_model_provider

logger = logging.getLogger(__name__)

_FUNDAMENTALS_KEYS = [
    "sector",
    "industry",
    "marketCap",
    "price",
    "currentPrice",
    "previousClose",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
    "revenueGrowth",
    "earningsGrowth",
    "debtToEquity",
    "targetMeanPrice",
    "recommendationKey",
    "numberOfAnalystOpinions",
    "52WeekChange",
    "6m_return_pct",
]

def _compact_fundamentals(f: dict) -> dict:
    if not isinstance(f, dict):
        return {}
    out = {k: f.get(k) for k in _FUNDAMENTALS_KEYS if k in f}
    if "price" not in out and "currentPrice" in out:
        out["price"] = out.get("currentPrice")
    return out


def _as_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip().replace("%", "").replace(",", "")
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _normalize_return_value(value) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    # Models often emit percent points for return fields: 10.4 means 10.4%.
    # Pipeline contracts store returns as decimals: 0.104.
    if abs(number) > 1.0:
        return number / 100.0
    return number


def _normalize_return_fields(obj) -> None:
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            key_l = str(key).lower()
            if key_l in {
                "probability_weighted_return_12m",
                "net_return_after_costs",
                "return_pct",
                "return_12m",
                "expected_return",
                "wdr",
            }:
                normalized = _normalize_return_value(value)
                if normalized is not None:
                    obj[key] = normalized
            else:
                _normalize_return_fields(value)
    elif isinstance(obj, list):
        for item in obj:
            _normalize_return_fields(item)


def _validate_scenarios_shape(result: dict) -> list[str]:
    """Validate normalized scenarios. Returns list of error strings (empty = valid)."""
    errors: list[str] = []
    scenarios = result.get("scenarios")
    if not isinstance(scenarios, dict):
        return ["scenarios missing or not dict"]
    required = {"bull", "base", "bear"}
    missing = required - set(scenarios.keys())
    if missing:
        errors.append(f"missing scenario keys: {sorted(missing)}")
    for key in required & set(scenarios.keys()):
        sc = scenarios.get(key)
        if not isinstance(sc, dict):
            errors.append(f"scenario {key!r} not dict")
            continue
        if sc.get("probability") is None:
            errors.append(f"scenario {key!r} missing probability")
        if sc.get("return_12m") is None and sc.get("target") is None and sc.get("expected_return") is None:
            errors.append(f"scenario {key!r} missing return/target")
    return errors


def _normalize_scenarios_structure(result: dict) -> None:
    scenarios = result.get("scenarios")
    if isinstance(scenarios, list):
        normalized = {}
        for item in scenarios:
            if not isinstance(item, dict):
                continue
            key = str(
                item.get("scenario_name")
                or item.get("name")
                or item.get("type")
                or item.get("label")
            ).strip().lower()
            if "bull" in key:
                normalized["bull"] = item
            elif "base" in key or "neutral" in key:
                normalized["base"] = item
            elif "bear" in key:
                normalized["bear"] = item
        if normalized:
            result["scenarios"] = normalized
    elif isinstance(scenarios, dict):
        normalized = {}
        for key in ("bull", "base", "bear"):
            if key in scenarios:
                normalized[key] = scenarios[key]
        for alt in ("bull_scenario", "base_scenario", "bear_scenario"):
            if alt in scenarios and alt.split("_")[0] not in normalized:
                normalized[alt.split("_")[0]] = scenarios[alt]
        if normalized:
            result["scenarios"] = normalized


def _normalize_scenario_result(
    result: dict,
    symbol: str = "",
    company_name: str = "",
    sector: str = "",
    fundamentals: dict | None = None,
) -> dict:
    """
    Normalize a raw scenario result dict in-place and return it.
    - Converts percent-point return values to decimals
    - Normalizes scenarios list → dict
    - Sets recommendation from EV if AVOID but EV is positive
    - Backfills current_price from fundamentals
    """
    _normalize_return_fields(result)
    _normalize_scenarios_structure(result)

    ev = _as_float(result.get("probability_weighted_return_12m"))
    if result.get("recommendation") in (None, "AVOID") and ev is not None and ev > 0:
        result["recommendation"] = _recommendation_from_ev(ev)

    if result.get("current_price") is None:
        price = _extract_current_price(fundamentals or {}, result)
        if price is not None:
            result["current_price"] = price

    if symbol:
        result.setdefault("symbol", symbol)
    if company_name:
        result.setdefault("company_name", company_name)
    if sector:
        result.setdefault("sector", sector)

    return result


def _recommendation_from_ev(ev: float | None) -> str:
    if ev is None:
        return "AVOID"
    if ev >= 0.12:
        return "STRONG_BUY"
    if ev >= 0.05:
        return "BUY"
    if ev > config.MIN_EXPECTED_RETURN:
        return "WATCH"
    return "AVOID"


def _extract_current_price(fundamentals: dict, result: dict) -> float | None:
    for source in (result, fundamentals):
        for key in ("current_price", "price", "currentPrice", "previousClose"):
            value = _as_float((source or {}).get(key))
            if value is not None and value > 0:
                return value
    return None


def _compute_ev_from_debate_scores(
    debate_scores: dict,
    bull_return: float,
    base_return: float,
    bear_return: float,
) -> float:
    """Compute EV using debate-derived probabilities."""
    p_bull = float(debate_scores.get("P_bull", 0.25))
    p_base = float(debate_scores.get("P_base", 0.50))
    p_bear = float(debate_scores.get("P_bear", 0.25))

    ev = (p_bull * bull_return) + (p_base * base_return) + (p_bear * bear_return)
    return ev


def _run_scenario_agent(args: tuple) -> dict:
    """Thread worker for scenario modeling."""
    symbol, company_name, sector, research_data, agent_id = args
    logger.info(f"  [Scenario Agent {agent_id:02d}] Modeling {symbol}...")

    debate_scores = research_data.get("debate_scores", {})
    synthesis_packet = research_data.get("synthesis_packet", {})
    bull_thesis = research_data.get("bull_thesis", "")
    bear_thesis = research_data.get("bear_thesis", "")

    try:
        provider, _ = parse_model_provider(config.STAGE3_MODEL)
        yf_symbol = f"{symbol}.NS"
        fundamentals = _compact_fundamentals(get_fundamentals(yf_symbol))

        # Build prompt that includes debate-derived probabilities
        # LLM only needs to set price targets, not probabilities
        system = (
            "You are a senior quantitative equity strategist.\n"
            "You are part of an autonomous pipeline focused on MAXIMUM PROFIT %.\n"
            "Your job: set PRICE TARGETS and DETAILED RETURNS for each scenario based on the Manager's guidance.\n"
            "Return valid JSON only. All returns must be decimals (0.104 means +10.4%).\n"
            "Maximize transparency and alpha-seeking logic."
        )
        agent = BaseAgent(
            system_prompt=system,
            tools=[],
            model=config.STAGE3_MODEL,
            max_tokens=config.STAGE3_AGENT_MAX_TOKENS,
        )

        ctx = {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "fundamentals": fundamentals,
            "research_packet": synthesis_packet,
            "bull_thesis_fallback": bull_thesis[:1200] if isinstance(bull_thesis, str) else "",
            "bear_thesis_fallback": bear_thesis[:1200] if isinstance(bear_thesis, str) else "",
            "debate_probabilities": {
                "P_bull": debate_scores.get("P_bull", 0.25),
                "P_base": debate_scores.get("P_base", 0.50),
                "P_bear": debate_scores.get("P_bear", 0.25),
                "bull_agents": debate_scores.get("bull_agents", []),
                "bear_agents": debate_scores.get("bear_agents", []),
            },
            "task": (
                "Build bull/base/bear scenarios using the probabilities and profit targets provided by the Portfolio Manager.\n"
                "You are a quant agent - refine these numbers with detailed bottom-up price target analysis.\n"
                "Return JSON with: scenarios.{bull,base,bear}.{probability,return_12m,price_targets,description},\n"
                "probability_weighted_return_12m, key_catalysts_next_90_days, key_risks,\n"
                "thesis_invalidation, investment_thesis, recommendation.\n"
                "All return fields MUST be decimals (0.104 = +10.4%).\n"
                "Your objective is to validate or refute the Manager's profit expectations with hard data."
            ),
        }

        raw = agent.run("Model scenarios using debate-derived probabilities.", context=ctx)
        try:
            result = extract_json(raw, expected=dict)
        except Exception as e:
            result = {
                "symbol": symbol, "company_name": company_name, "sector": sector,
                "error": str(e), "raw_response": raw[:500],
                "probability_weighted_return_12m": None, "recommendation": "AVOID",
            }

        _normalize_scenarios_structure(result)
        _normalize_return_fields(result)

        shape_errors = _validate_scenarios_shape(result)
        if shape_errors and not result.get("error"):
            result["error"] = "scenarios_malformed: " + "; ".join(shape_errors)
            result["recommendation"] = "AVOID"
            result["probability_weighted_return_12m"] = None

        # Scenario Agent refines the Manager's probabilities 
        # No more hard overrides - we trust the agentic chain!
        scenarios = result.get("scenarios", {})

        # Compute EV from debate probabilities × scenario returns
        bull_return = _normalize_return_value(
            (scenarios.get("bull") or {}).get("return_12m")
        ) or 0.0
        base_return = _normalize_return_value(
            (scenarios.get("base") or {}).get("return_12m")
        ) or 0.0
        bear_return = _normalize_return_value(
            (scenarios.get("bear") or {}).get("return_12m")
        ) or 0.0

        ev = _compute_ev_from_debate_scores(debate_scores, bull_return, base_return, bear_return)
        result["probability_weighted_return_12m"] = ev
        result["net_return_after_costs"] = ev - config.TOTAL_TRANSACTION_COST

        # Enrich result
        result["symbol"] = symbol
        result["company_name"] = company_name
        result["sector"] = sector
        result["debate_scores"] = debate_scores
        result["conviction"] = research_data.get("conviction", 0.5)

        # Price
        current_price = _extract_current_price(fundamentals, result)
        if current_price:
            result["current_price"] = current_price
            result["price"] = current_price

        # Recommendation from EV
        result["recommendation"] = _recommendation_from_ev(ev)

        # Debate log (traceability: which agents support which scenario)
        result["debate_log"] = {
            "bull_case": debate_scores.get("bull_agents", []),
            "bear_case": debate_scores.get("bear_agents", []),
            "hold_agents": debate_scores.get("hold_agents", []),
            "resolution": (
                f"Debate-derived: P_bull={debate_scores.get('P_bull', 0):.2f} "
                f"P_base={debate_scores.get('P_base', 0):.2f} "
                f"P_bear={debate_scores.get('P_bear', 0):.2f}. "
                f"Consensus={debate_scores.get('consensus_strength', 0):.2f}"
            ),
        }

        # Attach P_bull/P_bear/P_base at top level for validators
        result["P_bull"] = debate_scores.get("P_bull", 0.25)
        result["P_bear"] = debate_scores.get("P_bear", 0.25)
        result["P_base"] = debate_scores.get("P_base", 0.50)

        result["_token_usage"] = agent.get_token_usage()

        logger.info(
            f"  [Scenario Agent {agent_id:02d}] {symbol} complete — "
            f"EV: {ev*100:+.1f}% (bull={bull_return*100:+.1f}% base={base_return*100:+.1f}% bear={bear_return*100:+.1f}%)"
        )
        return result

    except Exception as e:
        logger.error(f"  [Scenario Agent {agent_id:02d}] {symbol} failed: {e}")
        return {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "error": str(e),
            "probability_weighted_return_12m": None,
            "recommendation": "AVOID",
            "debate_scores": debate_scores,
        }


def _apply_selection_pressure(results: list[dict]) -> list[dict]:
    """
    If >50% of stocks have positive EV, adjust the eligibility threshold.
    Only top 30% EV stocks proceed to portfolio stage (or those clearing dynamic threshold).
    Does NOT distort underlying debate probabilities.
    """
    valid = [r for r in results if r.get("probability_weighted_return_12m") is not None]
    if not valid:
        return results

    positive_ev = [r for r in valid if (r.get("probability_weighted_return_12m") or 0) > 0]
    positive_pct = len(positive_ev) / len(valid) if valid else 0

    dynamic_min_ev = config.MIN_EXPECTED_RETURN

    if config.EV_NORMALIZATION_ENABLED and positive_pct > 0.50:
        logger.warning(
            f"SELECTION PRESSURE: {len(positive_ev)}/{len(valid)} ({positive_pct*100:.0f}%) "
            f"have positive EV — increasing EV threshold to restrict eligibility."
        )

        valid.sort(key=lambda x: x.get("probability_weighted_return_12m", -999), reverse=True)
        target_positive_count = max(1, int(len(valid) * config.EV_POSITIVE_TARGET_PCT))
        
        # Set dynamic threshold to the EV of the marginal accepted stock
        if target_positive_count < len(valid):
            marginal_ev = valid[target_positive_count - 1].get("probability_weighted_return_12m", 0)
            # Add a slight buffer to the threshold
            dynamic_min_ev = max(config.MIN_EXPECTED_RETURN, marginal_ev)
            logger.info(f"SELECTION PRESSURE: Dynamic EV threshold set to {dynamic_min_ev*100:.2f}%")

    # Only stocks passing the dynamic threshold AND in the top N% proceed
    valid.sort(key=lambda x: x.get("probability_weighted_return_12m", -999), reverse=True)
    cutoff = max(config.MAX_POSITIONS, int(len(valid) * config.EV_TOP_PCT_TO_PORTFOLIO))
    
    for i, r in enumerate(valid):
        ev = r.get("probability_weighted_return_12m", -999)
        if i < cutoff and ev >= dynamic_min_ev:
            r["_portfolio_eligible"] = True
        else:
            r["_portfolio_eligible"] = False
            if ev > 0 and ev < dynamic_min_ev:
                r["_selection_pressure_applied"] = True # Mark it as rejected due to pressure

    advancing = [r for r in valid if r.get("_portfolio_eligible")]
    rejected = [r for r in valid if not r.get("_portfolio_eligible")]

    logger.info(
        f"SELECTION PRESSURE: {len(advancing)} stocks advance to portfolio, {len(rejected)} filtered out"
    )

    return valid + [r for r in results if r not in valid]


class Stage3ScenarioModeling:
    """
    Stage 3: Probability-weighted scenario modeling with debate-derived probabilities.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def run(
        self,
        stage2_research: dict[str, dict],
        top50: list[dict],
        save_path: Optional[Path] = None,
    ) -> list[dict]:
        """
        Run scenario modeling on all stocks from Stage 2.

        Args:
            stage2_research: Dict keyed by symbol with debate results
            top50: Stage 1 data (for sector info)
            save_path: Where to save results

        Returns:
            List of scenario model dicts, sorted by EV return (descending)
        """
        start_time = datetime.now()

        # Build stock metadata lookup
        stock_meta = {s["symbol"]: s for s in top50}

        # Prepare tasks
        tasks = []
        for i, (symbol, research) in enumerate(stage2_research.items()):
            meta = stock_meta.get(symbol, {})

            if not research.get("success", True):
                logger.warning(f"Skipping {symbol} — debate failed")
                continue

            tasks.append((
                symbol,
                research.get("company_name", meta.get("company_name", symbol)),
                research.get("sector", meta.get("sector", "Unknown")),
                research,  # Full research data including debate_scores
                i + 1,
            ))

        logger.info("=" * 60)
        logger.info(f"STAGE 3: Scenario Modeling ({len(tasks)} stocks)")
        logger.info(f"         Debate-derived probabilities + LLM price targets")
        logger.info(f"         Max parallel workers: {self.max_workers}")
        logger.info("=" * 60)

        # Run scenario agents in parallel
        scenario_results = []
        total_tokens = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_run_scenario_agent, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                scenario_results.append(result)
                usage = result.get("_token_usage", {})
                total_tokens += usage.get("total_tokens", 0)

        # Sort by probability-weighted EV return (descending)
        def sort_key(x):
            ev = x.get("probability_weighted_return_12m")
            return ev if ev is not None else -999

        scenario_results.sort(key=sort_key, reverse=True)

        # Add sector metadata from Stage 1 if missing
        for result in scenario_results:
            sym = result.get("symbol", "")
            if not result.get("sector") and sym in stock_meta:
                result["sector"] = stock_meta[sym].get("sector", "Unknown")

        # ── Apply selection pressure ──
        scenario_results = _apply_selection_pressure(scenario_results)

        # Stats
        valid = [r for r in scenario_results if r.get("probability_weighted_return_12m") is not None]
        positive_ev = [r for r in valid if r.get("probability_weighted_return_12m", 0) > 0]
        portfolio_eligible = [r for r in valid if r.get("_portfolio_eligible", True)]
        avg_ev = sum(r["probability_weighted_return_12m"] for r in positive_ev) / len(positive_ev) if positive_ev else 0

        duration = (datetime.now() - start_time).total_seconds()

        output = {
            "stage": 3,
            "run_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "total_modeled": len(scenario_results),
            "positive_ev_count": len(positive_ev),
            "portfolio_eligible_count": len(portfolio_eligible),
            "avg_positive_ev": avg_ev,
            "total_tokens": total_tokens,
            "selection_pressure_applied": any(r.get("_selection_pressure_applied") for r in scenario_results),
            "scenario_models": scenario_results,
            "debate_logs": [
                {
                    "symbol": r.get("symbol"),
                    "company_name": r.get("company_name"),
                    "debate_log": r.get("debate_log", {}),
                }
                for r in scenario_results
            ],
        }

        if save_path:
            with open(save_path, "w") as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"Stage 3 results saved to {save_path}")

        logger.info(f"\n✅ Stage 3 complete in {duration/60:.1f}min")
        logger.info(f"   Modeled: {len(scenario_results)} | Positive EV: {len(positive_ev)} | Portfolio eligible: {len(portfolio_eligible)}")
        logger.info(f"   Avg EV (positive-only): {avg_ev*100:+.1f}%")
        logger.info(f"   Total tokens: {total_tokens:,}")

        if valid:
            logger.info(f"\n   Top 10 by EV:")
            for r in valid[:10]:
                ev = r.get("probability_weighted_return_12m", 0)
                eligible = "✓" if r.get("_portfolio_eligible", True) else "✗"
                logger.info(
                    f"   {eligible} {r.get('symbol',''):12} {r.get('sector',''):15} "
                    f"EV: {ev*100:+6.1f}%  [{r.get('recommendation', '?')}] "
                    f"P_bull={r.get('P_bull', 0):.2f} P_bear={r.get('P_bear', 0):.2f}"
                )

        return scenario_results
