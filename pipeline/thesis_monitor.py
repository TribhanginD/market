"""
Thesis Monitor — runs daily between rebalancing cycles.
Scans last 24h news for each holding.
Flags thesis breaches that may require early rebalancing.
"""

from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Optional


import config
from agents.base_agent import ResearchAgent
from llm.json_utils import extract_json

logger = logging.getLogger(__name__)


MONITOR_SYSTEM_PROMPT = """You are a thesis integrity monitor for an Indian equity portfolio.

Your job is to scan recent news (last 24 hours) for each holding and determine:
1. Is the original investment thesis INTACT, PARTIALLY INTACT, or BREACHED?
2. Are there any emergency exit signals?

An INTACT thesis means: no material negative news that fundamentally changes the investment case.
A PARTIAL thesis means: there's some concerning news but the core thesis holds.
A BREACHED thesis means: there's news that invalidates the core reason for owning this stock.

Examples of thesis breaches:
- CEO fraud/resignation unexpectedly
- Major earnings miss (>20% below consensus)
- Regulatory action (sebi investigation, product recall)
- Key contract loss / major client departure
- Corporate governance failure
- Product recall or safety issue
- Credit rating downgrade to speculative grade

Return JSON for each stock:
```json
{
  "symbol": "SYMBOL",
  "company_name": "Name",
  "thesis_status": "INTACT | PARTIAL | BREACHED",
  "alert_level": "NONE | WATCH | URGENT",
  "original_thesis_summary": "One sentence",
  "news_summary": "What happened in the last 24 hours",
  "action_recommended": "HOLD | REVIEW | EXIT",
  "reason": "Why this action",
  "news_articles": [{"title": "...", "source": "...", "impact": "..."}]
}
```"""


class ThesisMonitor:
    """
    Daily thesis integrity monitor.
    Runs on all current holdings and flags any thesis breaches.
    """

    def __init__(self):
        self.agent = ResearchAgent(
            system_prompt=MONITOR_SYSTEM_PROMPT,
            model=config.MODEL_FAST,  # Use fast model for daily monitoring
            max_tokens=config.MAX_TOKENS,
        )

    def run(
        self,
        current_portfolio: dict,
        save_path: Optional[Path] = None,
    ) -> list[dict]:
        """
        Check all holdings for thesis integrity.
        
        Args:
            current_portfolio: Current portfolio.json with positions
            save_path: Where to save alerts
            
        Returns:
            List of thesis status dicts, one per holding
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("THESIS MONITOR: Daily Holdings Check")
        logger.info("=" * 60)
        
        positions = current_portfolio.get("positions", [])
        if not positions:
            logger.info("No current positions to monitor.")
            return []
        
        logger.info(f"Monitoring {len(positions)} holdings...")
        
        results = []
        urgent_alerts = []
        
        for i, position in enumerate(positions):
            symbol = position.get("symbol", "")
            company = position.get("company_name", symbol)
            original_thesis = position.get("rationale", "")
            
            logger.info(f"  [{i+1}/{len(positions)}] Checking {symbol}...")
            
            try:
                # Fresh agent for each stock (clean context)
                agent = ResearchAgent(
                    system_prompt=MONITOR_SYSTEM_PROMPT,
                    model=config.MODEL_FAST,
                    max_tokens=config.MAX_TOKENS,
                )
                
                prompt = f"""Check thesis integrity for holding: {symbol} ({company})

Original investment thesis:
{original_thesis or 'Not available — assess based on news only.'}

Instructions:
1. Call get_recent_news('{symbol}', '{company}', days={config.THESIS_MONITOR_LOOKBACK_DAYS}) to get last 24-48h news
2. Determine if thesis is INTACT, PARTIAL, or BREACHED
3. Return JSON with thesis status and recommended action"""
                
                raw = agent.run(prompt)
                
                # Parse result
                try:
                    status = extract_json(raw, expected=dict)
                except Exception:
                    logger.warning("Thesis monitor parse failed for %s — treating as UNKNOWN/WATCH", symbol)
                    status = {
                        "symbol": symbol,
                        "thesis_status": "UNKNOWN",
                        "alert_level": "WATCH",
                        "action_recommended": "HOLD",
                        "reason": "Monitor parse error — status unknown, review manually",
                        "error": "parse_failure",
                    }
                
                status["check_time"] = datetime.now().isoformat()
                results.append(status)
                
                if status.get("alert_level") == "URGENT":
                    urgent_alerts.append(status)
                    logger.warning(f"  ⚠️  URGENT ALERT: {symbol} — {status.get('reason', '')}")
                elif status.get("alert_level") == "WATCH":
                    logger.warning(f"  👁  WATCH: {symbol} — {status.get('reason', '')}")
                else:
                    logger.info(f"  ✅ {symbol}: thesis INTACT")
                    
            except Exception as e:
                logger.error(f"  Error monitoring {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "thesis_status": "UNKNOWN",
                    "alert_level": "NONE",
                    "action_recommended": "HOLD",
                    "error": str(e),
                })
        
        duration = (datetime.now() - start_time).total_seconds()
        
        output = {
            "monitor_date": datetime.now().strftime("%Y-%m-%d"),
            "run_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "total_checked": len(positions),
            "urgent_alerts": len(urgent_alerts),
            "watch_alerts": len([r for r in results if r.get("alert_level") == "WATCH"]),
            "all_intact": len(urgent_alerts) == 0,
            "results": results,
            "urgent": urgent_alerts,
        }
        
        if save_path:
            with open(save_path, "w") as f:
                json.dump(output, f, indent=2, default=str)
        
        # Save to thesis_alerts file
        try:
            alerts_file = config.THESIS_ALERTS_FILE
            existing = {}
            if alerts_file.exists():
                with open(alerts_file) as f:
                    existing = json.load(f)
            
            existing[datetime.now().strftime("%Y-%m-%d")] = {
                "urgent": urgent_alerts,
                "all_results": results,
            }
            
            with open(alerts_file, "w") as f:
                json.dump(existing, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not update thesis_alerts.json: {e}")

        try:
            from persistence import db as pdb
            pdb.append_thesis_checks(results)
        except Exception as e:
            logger.warning("DB write failed for thesis checks: %s", e)
        
        logger.info(f"\n✅ Thesis monitor complete in {duration:.0f}s")
        logger.info(f"   Checked: {len(positions)} | Urgent: {len(urgent_alerts)} | Watch: {output['watch_alerts']}")
        
        if urgent_alerts:
            logger.warning(f"\n⚠️  URGENT ALERTS ({len(urgent_alerts)}):")
            for alert in urgent_alerts:
                logger.warning(f"   {alert['symbol']}: {alert.get('reason', '')}")

            breached_symbols = [
                a["symbol"] for a in urgent_alerts
                if a.get("thesis_status") == "BREACHED" or a.get("alert_level") == "URGENT"
            ]
            if breached_symbols and getattr(config, "THESIS_MONITOR_AUTO_RERUN", True):
                logger.warning(
                    "\n   Auto-triggering partial pipeline rerun for: %s",
                    breached_symbols,
                )
                try:
                    from pipeline.orchestrator import run_partial_pipeline
                    partial_result = run_partial_pipeline(breached_symbols)
                    output["partial_rerun"] = {
                        "triggered_for": breached_symbols,
                        "run_id": partial_result.get("run_id"),
                        "status": "completed",
                    }
                    logger.info("   Partial rerun complete: run_id=%s", partial_result.get("run_id"))
                except Exception as e:
                    logger.error("   Partial rerun failed: %s", e)
                    output["partial_rerun"] = {
                        "triggered_for": breached_symbols,
                        "status": "failed",
                        "error": str(e),
                    }
            else:
                logger.warning("   Consider triggering early rebalance: python run.py --mode pipeline")

        return results
