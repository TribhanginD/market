"""
Stage 5: Rebalancing
Compares new pipeline output vs current holdings.
Generates specific BUY/SELL/HOLD decisions with full written rationale.
"""

from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Optional


import config
from agents.base_agent import BaseAgent
from llm.json_utils import extract_json

logger = logging.getLogger(__name__)


REBALANCE_SYSTEM_PROMPT = """You are the Master Rebalancing Orchestrator for an Indian AI equity portfolio.

Your job is to compare the CURRENT portfolio with the NEW optimal portfolio and generate specific trade recommendations with full written rationale.

TRADE CONDITIONS (debate-aware):
- BUY requires: risk_adjusted_ev > 5% AND consensus_strength > 0.5 (agents mostly agree)
  * Note: risk_adjusted_ev = EV - (0.5 * uncertainty_score)
- SELL requires: EV < 0% OR thesis invalidated by debate outcome
- HOLD: everything else

Rebalancing Rules:
1. ONLY recommend changes if the improvement is meaningful (new stock EV must be >5% higher than what it replaces)
2. Limit portfolio turnover: max 40% of portfolio can change per cycle
3. Factor in transaction costs (~0.15% per trade)
4. For every SELL: explain why the thesis has changed or a better opportunity exists
5. For every BUY: provide full thesis, EV, catalysts, and risks
6. HOLD decisions should also be listed with current status update
7. For every decision, include the strongest counterargument you considered and why it did not win

MANDATORY FIELDS FOR EVERY TRADE:
- "why_now": Specific near-term catalyst justifying entry timing
- "what_changed": Delta from last assessment (or "New position" for first-time BUYs)
- "what_would_make_this_wrong": Falsifiable condition for position exit

OUTPUT FORMAT — Return valid JSON:
```json
{
  "rebalance_date": "YYYY-MM-DD",
  "portfolio_unchanged": false,
  
  "trades": [
    {
      "action": "BUY",
      "symbol": "NEWSTOCK",
      "company_name": "Full Name",
      "sector": "Sector",
      "target_allocation_pct": 8.5,
      "target_allocation_inr": 85000,
      "current_price": 1234.50,
      "ev_12m_return": 0.28,
      "consensus_strength": 0.75,
      "rationale": "Full 2-3 sentence rationale: why buying, what the EV is, key catalysts in next 90 days, and main risks",
      "counterargument": "Strongest reason not to buy, and why it was outweighed",
      "entry_note": "Any specific entry timing (e.g., after Q4 results on May 10)",
      "why_now": "Specific near-term catalyst",
      "what_changed": "Delta from last assessment",
      "what_would_make_this_wrong": "Falsifiable exit condition"
    },
    {
      "action": "SELL",
      "symbol": "OLDSTOCK",
      "company_name": "Full Name",
      "sector": "Sector",
      "current_allocation_pct": 7.0,
      "sell_price_est": 890.00,
      "rationale": "Why selling: thesis change / better opportunity / sector rebalance",
      "counterargument": "Strongest reason to keep the stock, and why it was rejected",
      "holding_return_pct": 0.12,
      "why_now": "Specific trigger for exit",
      "what_changed": "What deteriorated since last assessment",
      "what_would_make_this_wrong": "What would need to happen to re-enter"
    },
    {
      "action": "HOLD",
      "symbol": "KEEPSTOCK",
      "company_name": "Full Name",
      "current_allocation_pct": 9.0,
      "new_target_allocation_pct": 9.5,
      "allocation_change_pct": 0.5,
      "thesis_status": "Intact / Partially intact / Weakening",
      "status_note": "Brief update on holding thesis",
      "counterargument": "Strongest reason to change allocation, and why it was rejected",
      "what_would_make_this_wrong": "Condition that would trigger a SELL"
    }
  ],
  
  "new_portfolio": [...],
  
  "rebalance_summary": {
    "buys": 3,
    "sells": 3,
    "holds": 12,
    "turnover_pct": 22.5,
    "expected_transaction_cost_pct": 0.15,
    "portfolio_ev_before": 0.18,
    "portfolio_ev_after": 0.23,
    "ev_improvement": 0.05
  },
  
  "rebalance_notes": "Key reasoning behind major decisions this cycle"
}
```"""


class Stage5Rebalancer:
    """
    Stage 5: Master rebalancing orchestrator.
    Generates trade list comparing current vs optimal portfolio.
    """

    def __init__(self):
        self.agent = BaseAgent(
            system_prompt=REBALANCE_SYSTEM_PROMPT,
            tools=[],
            model=config.STAGE5_MODEL,
            max_tokens=config.STAGE5_AGENT_MAX_TOKENS,
        )

    def run(
        self,
        new_portfolio: dict,
        current_portfolio: dict = None,
        scenario_models: list[dict] = None,
        save_path: Optional[Path] = None,
    ) -> dict:
        """
        Generate rebalancing recommendations.
        
        Args:
            new_portfolio: Stage 4 output (optimal portfolio)
            current_portfolio: Existing portfolio.json
            scenario_models: Stage 3 data (for EV context)
            save_path: Where to save output
            
        Returns:
            Rebalancing dict with trades log
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("STAGE 5: Rebalancing")
        logger.info("=" * 60)
        
        new_positions = new_portfolio.get("portfolio", [])
        current_positions = []
        
        if current_portfolio:
            current_positions = current_portfolio.get("positions", [])
        
        # If no current portfolio, this is the first run
        if not current_positions:
            logger.info("  First run — no existing portfolio. All positions are new BUYs.")
            result = self._first_run_trades(new_positions)
        else:
            # Compute diff and let Claude decide
            result = self._run_rebalancer(new_positions, current_positions, scenario_models)
        
        duration = (datetime.now() - start_time).total_seconds()
        result["stage"] = 5
        result["run_time"] = datetime.now().isoformat()
        result["duration_seconds"] = duration
        
        if save_path:
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Stage 5 results saved to {save_path}")
        
        trades = result.get("trades", [])
        buys = [t for t in trades if t.get("action") == "BUY"]
        sells = [t for t in trades if t.get("action") == "SELL"]
        holds = [t for t in trades if t.get("action") == "HOLD"]
        
        logger.info(f"\n✅ Stage 5 complete in {duration:.0f}s")
        logger.info(f"   Buys: {len(buys)} | Sells: {len(sells)} | Holds: {len(holds)}")
        
        if buys:
            logger.info(f"\n   📈 BUYS:")
            for b in buys:
                logger.info(f"     BUY  {b.get('symbol','')} @ {b.get('allocation_pct', b.get('target_allocation_pct', 0)):.1f}%  EV: {b.get('ev_12m_return', 0)*100:+.1f}%")
        if sells:
            logger.info(f"\n   📉 SELLS:")
            for s in sells:
                logger.info(f"     SELL {s.get('symbol','')} — {s.get('rationale', '')[:60]}")
        
        return result

    def _first_run_trades(self, new_positions: list[dict]) -> dict:
        """Generate BUY trades for initial portfolio creation."""
        trades = []
        skipped = []
        for pos in new_positions:
            debate_scores = pos.get("debate_scores", {})
            consensus = debate_scores.get("consensus_strength", 0.5)
            uncertainty = debate_scores.get("uncertainty_score", 0.0)
            ev = pos.get("ev_12m_return", 0)

            risk_adjusted_ev = ev - (0.5 * uncertainty)

            # Apply trade conditions: BUY only if risk_adjusted_ev > threshold AND consensus > threshold
            if risk_adjusted_ev < config.EV_BUY_THRESHOLD or consensus < config.CONSENSUS_BUY_THRESHOLD:
                skipped.append({
                    "symbol": pos.get("symbol"),
                    "ev": ev,
                    "risk_adjusted_ev": risk_adjusted_ev,
                    "consensus": consensus,
                    "reason": (
                        "low_risk_adj_ev" if risk_adjusted_ev < config.EV_BUY_THRESHOLD
                        else "low_consensus"
                    ),
                })
                continue

            trades.append({
                "action": "BUY",
                "symbol": pos.get("symbol"),
                "company_name": pos.get("company_name"),
                "sector": pos.get("sector"),
                "target_allocation_pct": pos.get("allocation_pct"),
                "target_allocation_inr": pos.get("allocation_inr"),
                "current_price": pos.get("current_price"),
                "ev_12m_return": ev,
                "risk_adjusted_ev": risk_adjusted_ev,
                "consensus_strength": consensus,
                "rationale": pos.get("position_rationale", "Initial portfolio construction."),
                "counterargument": pos.get("counterargument", ""),
                "entry_note": pos.get("entry_note", ""),
                "conviction": pos.get("conviction", "Medium"),
                "why_now": "Initial portfolio construction — first run.",
                "what_changed": "New position — no prior assessment.",
                "what_would_make_this_wrong": pos.get("exit_trigger", "Thesis breach"),
            })

        if not trades:
            logger.error(
                "First-run produced ZERO trades. EV_BUY_THRESHOLD=%s CONSENSUS_BUY_THRESHOLD=%s. Skipped=%s",
                config.EV_BUY_THRESHOLD, config.CONSENSUS_BUY_THRESHOLD, skipped,
            )
            raise RuntimeError(
                f"Stage 5 first-run filtered all {len(new_positions)} candidates. "
                f"No initial portfolio created. Tune EV_BUY_THRESHOLD/CONSENSUS_BUY_THRESHOLD or "
                f"investigate Stage 2/3 output quality."
            )

        if skipped:
            logger.info("First-run skipped %d candidates: %s", len(skipped), skipped)

        return {
            "rebalance_date": datetime.now().strftime("%Y-%m-%d"),
            "portfolio_unchanged": False,
            "is_initial_run": True,
            "trades": trades,
            "new_portfolio": new_positions,
            "rebalance_summary": {
                "buys": len(trades),
                "sells": 0,
                "holds": 0,
                "turnover_pct": 100.0,
                "is_initial": True,
            },
            "rebalance_notes": "Initial portfolio construction — debate-vetted positions only.",
        }

    def _run_rebalancer(
        self,
        new_positions: list[dict],
        current_positions: list[dict],
        scenario_models: list[dict] = None,
    ) -> dict:
        """Run rebalancer agent to diff current vs new portfolio."""
        
        # Compute symbol sets
        new_symbols = {p["symbol"] for p in new_positions}
        current_symbols = {p["symbol"] for p in current_positions}
        
        added_symbols = new_symbols - current_symbols
        removed_symbols = current_symbols - new_symbols
        kept_symbols = new_symbols & current_symbols
        
        logger.info(f"  New positions: {len(added_symbols)} | Removed: {len(removed_symbols)} | Kept: {len(kept_symbols)}")
        
        # Prepare context for agent
        new_portfolio_summary = []
        for p in new_positions:
            new_portfolio_summary.append({
                "symbol": p.get("symbol"),
                "company_name": p.get("company_name"),
                "sector": p.get("sector"),
                "target_pct": p.get("allocation_pct"),
                "ev_return": p.get("ev_12m_return"),
                "rationale": p.get("position_rationale"),
                "counterargument": p.get("counterargument", ""),
                "entry_note": p.get("entry_note", ""),
                "exit_trigger": p.get("exit_trigger", ""),
            })
        
        current_portfolio_summary = []
        for p in current_positions:
            current_portfolio_summary.append({
                "symbol": p.get("symbol"),
                "company_name": p.get("company_name"),
                "sector": p.get("sector"),
                "current_pct": p.get("allocation_pct") or p.get("target_allocation_pct"),
                "entry_price": p.get("entry_price"),
                "current_price": p.get("current_price"),
                "original_thesis": p.get("rationale", ""),
                "counterargument": p.get("counterargument", ""),
            })
        
        prompt = f"""Generate the rebalancing trade list for the Indian AI Portfolio.

CURRENT PORTFOLIO ({len(current_positions)} positions):
{json.dumps(current_portfolio_summary, indent=2, default=str)}

NEW OPTIMAL PORTFOLIO ({len(new_positions)} positions):
{json.dumps(new_portfolio_summary, indent=2, default=str)}

Symbol Changes:
- New entries (BUY): {sorted(added_symbols)}
- Exiting positions (SELL): {sorted(removed_symbols)}
- Continuing positions (review alloc): {sorted(kept_symbols)}

Hard Rules:
1. Max 40% turnover — if more than 40% would change, rank changes by EV improvement and only make the top changes
2. For allocation adjustments to held positions, only change if diff > 2%
3. Write specific, detailed rationale for every SELL explaining what changed
4. Write specific BUY rationale with EV numbers and catalysts for every new entry
5. HOLD entries need a brief thesis status update

Return valid JSON following the specified output format."""
        
        raw = self.agent.run(prompt)
        
        try:
            result = extract_json(raw, expected=dict)
        except Exception as e:
            result = {"trades": [], "error": f"Parse failed: {e}", "raw": raw[:500]}
        
        return result
