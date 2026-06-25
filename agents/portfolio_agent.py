"""
Portfolio Agent — the "Agent of Agents".
Selects exactly 15 positions from scenario-modeled candidates,
enforcing all hard constraints, and writes full position rationale.
"""

import json

from agents.base_agent import BaseAgent
import config
from llm.json_utils import extract_json
from llm.providers import parse_model_provider


def _normalize_scenarios_for_portfolio(scenarios):
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
        return normalized or scenarios
    if isinstance(scenarios, dict):
        normalized = {}
        for key in ("bull", "base", "bear"):
            if key in scenarios:
                normalized[key] = scenarios[key]
        for alt in ("bull_scenario", "base_scenario", "bear_scenario"):
            if alt in scenarios and alt.split("_")[0] not in normalized:
                normalized[alt.split("_")[0]] = scenarios[alt]
        return normalized or scenarios
    return scenarios


PORTFOLIO_SYSTEM_PROMPT = f"""You are the Chief Investment Officer (CIO) of an AI-driven Indian equity fund.
Your job is to select the optimal 15-stock portfolio from a list of candidates that have already been
researched, debated (bull vs bear), and scenario-modeled.

HARD CONSTRAINTS (non-negotiable):
1. Exactly {config.MAX_POSITIONS} positions
2. No sector > {int(config.MAX_SECTOR_PCT * 100)}% of portfolio
3. Every position MUST have positive probability-weighted expected return
4. Portfolio aggregate 12-month expected return must beat Nifty 50 (target: >{int(config.BENCHMARK_ANNUAL_RETURN_TARGET * 100)}%)
5. Max position size: {int(config.MAX_POSITION_PCT * 100)}%
6. Min position size: {int(config.MIN_POSITION_PCT * 100)}%
7. Long-only (no shorts, no derivatives)
8. Account for India transaction costs: ~0.15% round trip per trade

OPTIMIZATION CRITERIA (in priority order):
1. Maximize probability-weighted expected return
2. Maximize sector diversification
3. Prefer higher confidence scores (>7/10)
4. Lower beta stocks get slight allocation boost for risk management
5. Prefer stocks with concrete near-term catalysts (next 90 days)

OUTPUT FORMAT — Return a specific JSON object:
```json
{{
  "portfolio": [
    {{
      "rank": 1,
      "symbol": "SYMBOL",
      "company_name": "Full Name",
      "sector": "Sector",
      "allocation_pct": 12.5,
      "allocation_inr": 125000,
      "current_price": 1234.50,
      "ev_12m_return": 0.28,
      "conviction": "High/Medium/Low",
      "position_rationale": "2-3 sentences on WHY this specific position, size, and conviction. Include expected return, key catalysts, and why it beats alternatives.",
      "entry_note": "Any specific entry timing notes (after earnings, wait for dip, etc.)",
      "exit_trigger": "The single event that would make you exit"
    }}
  ],
  
  "portfolio_summary": {{
    "total_positions": 15,
    "total_allocation_pct": 100.0,
    "weighted_avg_ev_return": 0.24,
    "vs_nifty50_alpha": 0.12,
    "sector_breakdown": {{"Financials": 25, "Technology": 20}},
    "top_3_positions_pct": 40.0,
    "portfolio_beta": 0.95
  }},
  
  "excluded_stocks": [
    {{
      "symbol": "EXCLUDED",
      "reason": "Why excluded despite high EV (e.g., sector concentration, low confidence)"
    }}
  ],
  
  "construction_notes": "Key decisions made during portfolio construction — trade-offs, constraints that bound the solution, alternative approaches considered"
}}
```

Be precise. Every allocation must be justified. The numbers must sum to 100%."""


class PortfolioAgent(BaseAgent):
    """Portfolio construction optimizer — selects 15 stocks with allocations."""

    def __init__(self):
        provider, _ = parse_model_provider(config.STAGE4_MODEL)
        system = PORTFOLIO_SYSTEM_PROMPT
        if provider in ("openai", "gemini", "groq"):
            # Compact system prompt to fit small context windows.
            system = (
                "You are a portfolio optimizer for Indian equities.\n"
                "Pick exactly 15 long-only stocks from candidates.\n"
                "Constraints: sector<=35%, position<=15%, position>=3%, all EV_12m>0, sum alloc=100.\n"
                "Before choosing, explicitly weigh the strongest counterargument for each top candidate and why it was rejected.\n"
                "Return JSON only: {portfolio:[{rank,symbol,sector,allocation_pct,allocation_inr,current_price,ev_12m_return,conviction,position_rationale,exit_trigger}],"
                "portfolio_summary:{total_positions,total_allocation_pct,weighted_avg_ev_return,sector_breakdown},excluded_stocks:[{symbol,reason}],construction_notes}.\n"
                "Keep rationales short but include why the position survived debate."
            )
        super().__init__(
            system_prompt=system,
            tools=[],  # No external tools — works from scenario data
            model=config.STAGE4_MODEL,
            max_tokens=config.STAGE4_AGENT_MAX_TOKENS,
        )

    def construct(
        self,
        scenario_models: list[dict],
        current_portfolio: dict = None,
        macro_context: dict = None,
    ) -> dict:
        """
        Select 15 positions from scenario-modeled candidates.
        
        Args:
            scenario_models: List of scenario dicts from Stage 3
            current_portfolio: Current holdings (for rebalancing context)
            macro_context: Current macro environment
            
        Returns:
            Portfolio construction dict
        """
        portfolio_value = config.PAPER_PORTFOLIO_VALUE_INR
        
        provider, _ = parse_model_provider(config.STAGE4_MODEL)

        # Prepare candidate summary for the agent
        candidates_summary = []
        for model in scenario_models:
            if not model.get("probability_weighted_return_12m"):
                continue
            if model.get("error"):
                continue

            scenarios = _normalize_scenarios_for_portfolio(model.get("scenarios"))
            if not isinstance(scenarios, dict):
                scenarios = {}

            if provider in ("openai", "gemini", "groq"):
                # Compact payload to fit small context windows.
                catalysts = model.get("key_catalysts_next_90_days", []) or []
                risks = model.get("key_risks", []) or []
                candidates_summary.append(
                    {
                        "symbol": model.get("symbol"),
                        "sector": model.get("sector", "Unknown"),
                        "price": model.get("current_price"),
                        "ev_12m": model.get("probability_weighted_return_12m"),
                        "confidence": model.get("confidence_score"),
                        "bull_p": scenarios.get("bull", {}).get("probability"),
                        "bear_p": scenarios.get("bear", {}).get("probability"),
                        "catalysts": [str(x)[:80] for x in catalysts[:3]],
                        "risks": [str(x)[:80] for x in risks[:3]],
                    }
                )
            else:
                candidates_summary.append({
                    "symbol": model.get("symbol"),
                    "company_name": model.get("company_name"),
                    "sector": model.get("sector", "Unknown"),
                    "current_price": model.get("current_price"),
                    "ev_return_12m": model.get("probability_weighted_return_12m"),
                    "net_return_after_costs": model.get("net_return_after_costs"),
                    "recommendation": model.get("recommendation"),
                    "confidence_score": model.get("confidence_score"),
                    "bull_probability": scenarios.get("bull", {}).get("probability"),
                    "bear_probability": scenarios.get("bear", {}).get("probability"),
                    "key_catalysts": model.get("key_catalysts_next_90_days", []),
                    "key_risks": model.get("key_risks", []),
                    "thesis": model.get("investment_thesis", ""),
                    "thesis_invalidation": model.get("thesis_invalidation", ""),
                    "analyst_consensus": model.get("analyst_consensus", {}),
                })
        # Small-context endpoints: keep only a small candidate set to fit prompt budget.
        if provider in ("openai", "gemini", "groq") and len(candidates_summary) > 12:
            candidates_summary = candidates_summary[:12]
        
        current_holdings_context = ""
        if provider not in ("openai", "gemini", "groq"):
            if current_portfolio and current_portfolio.get("positions"):
                current_holdings_context = f"""
CURRENT PORTFOLIO (for context — you are doing a full fresh selection):
Current holdings: {json.dumps(current_portfolio.get('positions', []), separators=(',', ':'), default=str)}
Note: The rebalancer (Stage 5) will handle the diff — just select the optimal portfolio."""
        
        macro_context_str = ""
        if macro_context:
            if provider in ("openai", "gemini", "groq"):
                macro_context_str = f"""
MACRO:
USD/INR {macro_context.get('usd_inr', 'N/A')} | VIX {macro_context.get('india_vix', 'N/A')} | Nifty1m {macro_context.get('nifty50_1m_return', 'N/A')}"""
            else:
                macro_context_str = f"""
CURRENT MACRO ENVIRONMENT:
- Nifty 50 1-month return: {macro_context.get('nifty50_1m_return', 'N/A')}
- Nifty 50 3-month return: {macro_context.get('nifty50_3m_return', 'N/A')}
- USD/INR: {macro_context.get('usd_inr', 'N/A')}
- India VIX: {macro_context.get('india_vix', 'N/A')}
- Top sector: {max(macro_context.get('sector_1m_returns', {}).items(), key=lambda x: x[1], default=('N/A', 0))[0] if macro_context.get('sector_1m_returns') else 'N/A'}"""
        
        candidates_json = json.dumps(candidates_summary, separators=(",", ":"), default=str)
        if provider in ("openai", "gemini", "groq"):
            prompt = (
                f"Build 15-stock portfolio. Value INR {portfolio_value}. "
                f"Constraints: sector<=35, pos<=15, pos>=3, sum=100, EV>0. {macro_context_str}\n"
                f"CANDIDATES={candidates_json}\n"
                "Return JSON only."
            )
        else:
            prompt = f"""Construct the optimal 15-stock Indian equity portfolio from the following {len(candidates_summary)} candidates.

Portfolio value to allocate: ₹{portfolio_value:,.0f}
Benchmark to beat: Nifty 50 (target >12% alpha)
{macro_context_str}
{current_holdings_context}

CANDIDATE STOCKS (sorted by probability-weighted 12m expected return):
{candidates_json}

Task:
1. Review all {len(candidates_summary)} candidates carefully
2. Select the 15 that maximize portfolio EV while respecting ALL hard constraints
3. Assign precise % allocations (must sum to 100%)
4. Calculate INR allocation for each position (based on ₹{portfolio_value:,.0f} total)
5. Write a 2-3 sentence rationale for each position explaining: why THIS stock vs alternatives, the strongest counterargument you considered, the specific EV, the key catalyst, and exit trigger
6. Note key excluded stocks and why
7. Return valid JSON following the exact output format specified

Remember: Every position must have positive EV. No sector can exceed 35%."""

        raw_response = self.run(prompt)
        
        # Parse JSON
        try:
            return extract_json(raw_response, expected=dict)
        except Exception:
            return {
                "error": "Failed to parse portfolio construction output",
                "raw_response": raw_response[:1000],
                "portfolio": [],
            }
