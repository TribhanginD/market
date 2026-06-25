"""
Stage 2: Multi-Agent Adversarial Debate

Replaces the old bull/bear parallel system with a 3-round interactive debate:
  Round 1: 4 agents (Growth, Value, Macro, Risk) produce independent theses
  Round 2: Each agent reads all others and rebuts the weakest claim
  Round 3: Each agent updates stance and confidence

Output: debate transcript + deterministic scores for Stage 3.
"""

from pathlib import Path
import json
import logging
import time
import concurrent.futures
from datetime import datetime
from typing import Optional


import config
from agents.debate_engine import DebateEngine
from agents.agent_memory import load_agent_weights, record_predictions


def _make_engine():
    """Pick debate engine per config.DEBATE_ENGINE: 'langgraph' (default) or 'ag2'."""
    engine_name = (getattr(config, "DEBATE_ENGINE", "langgraph") or "langgraph").lower()
    if engine_name == "ag2":
        from agents.ag2_debate import AG2DebateEngine
        return AG2DebateEngine(max_workers=config.DEBATE_MAX_WORKERS)
    return DebateEngine(max_workers=config.DEBATE_MAX_WORKERS)

from data.fetcher import get_fundamentals, get_news_for_stock, get_macro_context
from data.analyst_reports import get_analyst_reports_for_stock
from data.broker_feeds import get_research_signals

logger = logging.getLogger(__name__)


_FUNDAMENTALS_KEYS = [
    "symbol",
    "shortName",
    "longName",
    "sector",
    "industry",
    "currency",
    "exchange",
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

def _compact_news(news: list[dict], max_items: int = 8) -> list[dict]:
    if not isinstance(news, list):
        return []
    out = []
    for a in news[:max_items]:
        if not isinstance(a, dict):
            continue
        out.append(
            {
                "title": a.get("title"),
                "source": a.get("source"),
                "published_at": a.get("published_at") or a.get("date"),
                "summary": (a.get("summary") or "")[:240],
            }
        )
    return out

def _compact_macro(m: dict) -> dict:
    if not isinstance(m, dict):
        return {}
    keys = [
        "as_of",
        "nifty50_current",
        "nifty50_1m_return",
        "nifty50_3m_return",
        "india_vix",
        "usd_inr",
        "brent_oil",
    ]
    return {k: m.get(k) for k in keys if k in m}


def _format_fundamentals_summary(stock: dict) -> str:
    """Format a compact fundamentals summary for agent context."""
    return f"""
Symbol: {stock.get('symbol')} | Price: ₹{stock.get('price', 'N/A')}
PE: {stock.get('trailing_pe', 'N/A')} | PB: {stock.get('priceToBook', 'N/A')}
ROE: {stock.get('roe', 'N/A')} | Revenue Growth: {stock.get('revenue_growth', 'N/A')}
52W Return: {stock.get('return_52w', 'N/A')} | 6M Return: {stock.get('return_6m', 'N/A')}
Market Cap: ₹{stock.get('market_cap', 'N/A')}
Analyst Rec: {stock.get('analyst_recommendation', 'N/A')} | Analysts: {stock.get('num_analysts', 'N/A')}
""".strip()


def _prefetch_stock_data(stock: dict) -> dict:
    """Pre-fetch all data for a stock before the debate. No tool calls during debate."""
    symbol = stock.get("symbol", "")
    company_name = stock.get("company_name", symbol)
    yf_symbol = stock.get("yf_symbol") or f"{symbol}.NS"

    fundamentals = _compact_fundamentals(get_fundamentals(yf_symbol))
    news = _compact_news(
        get_news_for_stock(symbol, company_name, days=config.STAGE2_RESEARCH_LOOKBACK_DAYS)
    )
    macro = _compact_macro(get_macro_context())
    analyst_reports = get_analyst_reports_for_stock(symbol, company_name, days=14)[:6]

    current_price = fundamentals.get("price") or fundamentals.get("currentPrice")
    try:
        research_signals = get_research_signals(yf_symbol, current_price=current_price)
    except Exception as e:
        logger.warning("research signals fetch failed for %s: %s", symbol, e)
        research_signals = {"broker_actions": {}, "earnings_revisions": {}}

    return {
        "fundamentals": fundamentals,
        "recent_news": news,
        "macro": macro,
        "analyst_reports": analyst_reports,
        "broker_actions": research_signals.get("broker_actions", {}),
        "earnings_revisions": research_signals.get("earnings_revisions", {}),
        "fundamentals_summary": _format_fundamentals_summary(stock),
    }


def _run_debate_for_stock(args: tuple) -> tuple[str, dict]:
    """Thread worker to run a full debate for one stock."""
    stock, agent_weights = args
    symbol = stock.get("symbol", "")
    company_name = stock.get("company_name", symbol)
    sector = stock.get("sector", "Unknown")

    logger.info(f"\n{'═'*60}")
    logger.info(f"  Starting debate for {symbol} ({company_name})")
    logger.info(f"{'═'*60}")

    try:
        # Pre-fetch all data
        data_context = _prefetch_stock_data(stock)

        # Run 3-round debate (engine selected by config.DEBATE_ENGINE)
        engine = _make_engine()
        transcript = engine.run_debate(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            data_context=data_context,
        )

        # Extract exact debate scores generated autonomously by the Portfolio Manager node
        debate_scores = transcript.get("debate_scores", {"P_bull": 0.33, "P_bear": 0.33, "P_base": 0.34})
        conviction = transcript.get("conviction", 0.5)

        # Build result
        result = {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "price": data_context.get("fundamentals", {}).get("price")
                     or data_context.get("fundamentals", {}).get("currentPrice"),
            "trailing_pe": data_context.get("fundamentals", {}).get("trailingPE"),
            "roe": data_context.get("fundamentals", {}).get("returnOnEquity"),
            "revenue_growth": data_context.get("fundamentals", {}).get("revenueGrowth"),
            "earnings_growth": data_context.get("fundamentals", {}).get("earningsGrowth"),
            "debate_transcript": transcript,
            "debate_scores": debate_scores,
            "conviction": conviction,
            "token_usage": transcript.get("token_usage", {}),
            "success": True,
        }

        # Build legacy-compatible bull/bear thesis summaries for Stage 3
        r1 = transcript.get("round1", [])
        bull_points = []
        bear_points = []
        for agent_out in r1:
            stance = (agent_out.get("stance") or "").upper()
            args_list = agent_out.get("arguments", [])
            agent_type = agent_out.get("agent_type", "")
            for arg in args_list:
                point = f"[{agent_type.upper()}] {arg.get('point', '')}"
                if stance == "BUY":
                    bull_points.append(point)
                elif stance == "SELL":
                    bear_points.append(point)
                else:
                    # HOLD arguments go to both sides
                    bull_points.append(point)
                    bear_points.append(point)

        result["bull_thesis"] = "\n".join(bull_points[:10]) or "No bull arguments."
        result["bear_thesis"] = "\n".join(bear_points[:10]) or "No bear arguments."
        result["bull_success"] = True
        result["bear_success"] = True

        # Synthesis packet for Stage 3 (compact)
        result["synthesis_packet"] = {
            "symbol": symbol,
            "asof": datetime.now().strftime("%Y-%m-%d"),
            "bull_points": [p[:200] for p in bull_points[:5]],
            "bear_points": [p[:200] for p in bear_points[:5]],
            "key_uncertainties": [],
            "debate_log": {
                "bull_summary": [f"{a.get('agent_type')}: {a.get('stance')}" for a in r1 if (a.get("stance") or "").upper() == "BUY"],
                "bear_summary": [f"{a.get('agent_type')}: {a.get('stance')}" for a in r1 if (a.get("stance") or "").upper() == "SELL"],
                "resolution": f"Debate scored: P_bull={debate_scores['P_bull']:.2f}, P_bear={debate_scores['P_bear']:.2f}, P_base={debate_scores['P_base']:.2f}",
            },
            "numeric_facts": {
                "price": result.get("price"),
                "pe": result.get("trailing_pe"),
                "roe": result.get("roe"),
                "revenue_growth": result.get("revenue_growth"),
                "earnings_growth": result.get("earnings_growth"),
            },
            "catalysts_90d": [],
            "risks": [],
        }

        logger.info(
            f"  Debate complete for {symbol}: "
            f"P_bull={debate_scores['P_bull']:.2f} P_base={debate_scores['P_base']:.2f} "
            f"P_bear={debate_scores['P_bear']:.2f} conviction={conviction:.2f}"
        )

        return symbol, result

    except Exception as e:
        logger.error(f"  Debate failed for {symbol}: {e}")
        return symbol, {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "success": False,
            "error": str(e),
            "bull_success": False,
            "bear_success": False,
            "debate_scores": {"P_bull": 0.25, "P_bear": 0.25, "P_base": 0.50},
            "conviction": 0.0,
        }


class Stage2AdversarialResearch:
    """
    Stage 2: Multi-agent adversarial debate system.
    Runs 3-round debates for each stock with 4 specialized agents.
    """

    def __init__(self, max_workers: int = 6):
        self.max_workers = max_workers

    def run(
        self,
        top50: list[dict],
        n_stocks: int = None,
        run_id: str = "",
        save_path: Optional[Path] = None,
    ) -> dict[str, dict]:
        """
        Run adversarial debates on top N stocks.

        Args:
            top50: Stage 1 output
            n_stocks: Number of stocks to analyze (default: STAGE2_TOP_N)
            save_path: Where to save results JSON

        Returns:
            Dict keyed by symbol with debate results
        """
        n_stocks = n_stocks or config.STAGE2_TOP_N
        candidates = top50[:n_stocks]

        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"STAGE 2: Multi-Agent Adversarial Debate ({len(candidates)} stocks)")
        logger.info(f"         Agents: {', '.join(a.upper() for a in config.DEBATE_AGENTS)}")
        logger.info(f"         Rounds: {config.DEBATE_ROUNDS}")
        logger.info(f"         Max parallel workers: {self.max_workers}")
        logger.info("=" * 60)

        # Load agent weights from memory (for debate scoring)
        agent_weights = {}
        if getattr(config, "AGENT_MEMORY_ENABLED", True):
            try:
                agent_weights = load_agent_weights()
                logger.info(f"  Agent weights loaded: {agent_weights}")
            except Exception as e:
                logger.warning(f"  Failed to load agent weights: {e}")

        results_by_symbol: dict[str, dict] = {}
        total_tokens = 0

        # Run debates — one per stock, parallelized across stocks
        tasks = [(stock, agent_weights) for stock in candidates]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_debate_for_stock, task): task[0].get("symbol", "?")
                for task in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                symbol, result = future.result()
                results_by_symbol[symbol] = result
                usage = result.get("token_usage", {})
                total_tokens += int(usage.get("total_tokens") or 0)

                if config.DEBATE_BETWEEN_STOCKS_SLEEP > 0:
                    time.sleep(config.DEBATE_BETWEEN_STOCKS_SLEEP)

        # Record predictions to agent memory
        if run_id and getattr(config, "AGENT_MEMORY_ENABLED", True):
            try:
                debate_transcripts = [
                    r.get("debate_transcript", {})
                    for r in results_by_symbol.values()
                    if r.get("success") and r.get("debate_transcript")
                ]
                record_predictions(run_id, debate_transcripts)
            except Exception as e:
                logger.warning(f"Failed to record agent predictions: {e}")

        # Persist artifacts to SQLite
        if run_id:
            try:
                from persistence import db as pdb
                for sym, record in results_by_symbol.items():
                    artifacts = []
                    transcript = record.get("debate_transcript", {})
                    for r in transcript.get("round1", []):
                        artifacts.append({
                            "kind": f"debate_r1_{r.get('agent_type', '')}",
                            "agent_id": None,
                            "payload": r,
                        })
                    for r in transcript.get("round2", []):
                        artifacts.append({
                            "kind": f"debate_r2_{r.get('agent_type', '')}",
                            "agent_id": None,
                            "payload": r,
                        })
                    for r in transcript.get("round3", []):
                        artifacts.append({
                            "kind": f"debate_r3_{r.get('agent_type', '')}",
                            "agent_id": None,
                            "payload": r,
                        })
                    artifacts.append({
                        "kind": "debate_scores",
                        "agent_id": None,
                        "payload": record.get("debate_scores", {}),
                    })
                    pdb.append_research_artifacts(run_id, sym, artifacts)
            except Exception:
                pass

        duration = (datetime.now() - start_time).total_seconds()

        output = {
            "stage": 2,
            "run_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "stocks_analyzed": len(candidates),
            "total_tokens": total_tokens,
            "debate_system": {
                "agents": config.DEBATE_AGENTS,
                "rounds": config.DEBATE_ROUNDS,
                "agent_weights": agent_weights,
            },
            "research": results_by_symbol,
        }

        if save_path:
            with open(save_path, "w") as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"Stage 2 results saved to {save_path}")

        successes = sum(1 for s in results_by_symbol.values() if s.get("success"))
        logger.info(f"\n✅ Stage 2 complete in {duration/60:.1f}min")
        logger.info(f"   Stocks debated: {len(candidates)} | Successful: {successes}")
        logger.info(f"   Total tokens used: {total_tokens:,}")

        return results_by_symbol

    @staticmethod
    def _combine_cases(label: str, cases: list[dict]) -> str:
        """Legacy compatibility method."""
        successful = [case for case in cases if case.get("success") and case.get("thesis")]
        if not successful:
            return f"{label} research unavailable."
        combined = []
        for case in sorted(successful, key=lambda item: item.get("agent_id", 0)):
            combined.append(f"[{label} AGENT {case.get('agent_id')}]\n{case.get('thesis')}")
        return "\n\n".join(combined)

    @staticmethod
    def _debate_summary(cases: list[dict], label: str) -> list[str]:
        """Legacy compatibility method."""
        out = []
        for case in sorted([c for c in cases if c.get("success") and c.get("thesis")], key=lambda item: item.get("agent_id", 0)):
            thesis = str(case.get("thesis") or "").strip()
            if thesis:
                out.append(f"{label} {case.get('agent_id')}: {thesis[:500]}")
        return out[:6]

    @staticmethod
    def _debate_resolution(packet: dict, bull_cases: list[dict], bear_cases: list[dict]) -> str:
        """Legacy compatibility method."""
        if not isinstance(packet, dict):
            return "No synthesis packet."
        bulls = len([c for c in bull_cases if c.get("success")])
        bears = len([c for c in bear_cases if c.get("success")])
        q = packet.get("key_uncertainties") or []
        if bulls and bears:
            return f"Adjudicated {bulls} bull vs {bears} bear views; uncertainties: {', '.join(map(str, q[:3]))}"
        return "Incomplete debate."
