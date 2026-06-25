"""
Stage 1: Universe Screening
Scores all Nifty 500 stocks on fundamentals + momentum + macro fit.
Output: Top 50 ranked stocks that advance to Stage 2.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


import config
from data.nifty500 import get_nifty500
from data.fetcher import get_fundamentals_batch, compute_composite_score, get_macro_context, get_live_prices_batch
from agents.base_agent import BaseAgent
from llm.json_utils import extract_json
from llm.providers import parse_model_provider

logger = logging.getLogger(__name__)


SCREENING_SYSTEM_PROMPT = """You are a quantitative equity analyst screening the Nifty 500 for the best investment opportunities.

You will receive a pre-scored list of stocks with composite scores from a quantitative model.
Your job is to:
1. Review the top candidates and apply qualitative adjustments
2. Flag any stocks that should be excluded (suspended trading, corporate fraud allegations, promoter pledge > 50%, etc.)
3. Identify any obvious omissions — undervalued stocks the quant model may have missed
4. Return the final ranked top 50 list with brief reasoning

Return a JSON array of the top 50 stocks in this format:
```json
[
  {
    "rank": 1,
    "symbol": "SYMBOL",
    "company_name": "Name",
    "sector": "Sector",
    "composite_score": 82.5,
    "quant_rank": 1,
    "analyst_rank": 1,
    "advance_reason": "One sentence on why this stock advances — what makes it stand out",
    "flags": []
  }
]
```

Be decisive. Return exactly 50 stocks."""


class Stage1Screener:
    """
    Stage 1: Quantitative + qualitative screening of Nifty 500.
    """

    def __init__(self):
        provider, _ = parse_model_provider(config.STAGE1_MODEL)
        # Gemini/Groq are used in "no-tools" mode; for test runs we skip LLM review here to avoid
        # transient 503s and strict JSON brittleness on a huge prompt.
        self.use_llm_review = bool(config.STAGE1_USE_LLM_REVIEW) and provider == "anthropic"
        self.agent = BaseAgent(
            system_prompt=SCREENING_SYSTEM_PROMPT,
            tools=[],
            model=config.STAGE1_MODEL,
            max_tokens=config.MAX_TOKENS,
        )

    def run(self, save_path: Optional[Path] = None) -> list[dict]:
        """
        Run Stage 1 screening.
        
        Returns:
            List of top 50 stock dicts, sorted by score
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("STAGE 1: Universe Screening")
        logger.info("=" * 60)
        
        # Step 1: Get Nifty 500 universe
        logger.info("Fetching Nifty 500 universe...")
        universe_df = get_nifty500()
        if getattr(config, "REQUIRE_FULL_UNIVERSE", False) and len(universe_df) < getattr(config, "MIN_UNIVERSE_SIZE", 500):
            raise RuntimeError(
                f"Universe size too small ({len(universe_df)}). "
                "Refusing to run with partial universe."
            )
        logger.info(f"Universe: {len(universe_df)} stocks across {universe_df['sector'].nunique()} sectors")
        
        # Step 2: Fetch fundamentals for all stocks (cached)
        logger.info(f"Fetching fundamentals for {len(universe_df)} stocks (cached where available)...")
        yf_symbols = universe_df["yf_symbol"].tolist()

        logger.info("Fetching live prices (yfinance history, batched)...")
        live_prices = get_live_prices_batch(yf_symbols)
        ok_live = sum(1 for v in live_prices.values() if (v or {}).get("price") is not None)
        live_cov = ok_live / max(1, len(yf_symbols))
        min_cov = getattr(config, "MIN_FUNDAMENTALS_COVERAGE_PCT", 0.0)
        if live_cov < min_cov:
            raise RuntimeError(
                f"Live price coverage too low ({live_cov:.1%} < {min_cov:.1%}). "
                "Fix yfinance symbol mapping / connectivity before running."
            )

        all_fundamentals = get_fundamentals_batch(yf_symbols, delay_seconds=0.2)

        ok_fund = 0
        for yf_sym in yf_symbols:
            fund = all_fundamentals.get(yf_sym, {})
            has_price = bool(fund.get("price") or fund.get("currentPrice") or fund.get("previousClose"))
            if "error" not in fund and has_price:
                ok_fund += 1
        coverage = ok_fund / max(1, len(yf_symbols))
        min_cov = getattr(config, "MIN_FUNDAMENTALS_COVERAGE_PCT", 0.0)
        if coverage < min_cov:
            raise RuntimeError(
                f"Fundamentals coverage too low ({coverage:.1%} < {min_cov:.1%}). "
                "Fix live price/fundamentals feed before running."
            )
        
        # Step 3: Quantitative scoring
        logger.info("Computing quantitative composite scores...")
        scored_stocks = []
        
        for _, row in universe_df.iterrows():
            sym = row["symbol"]
            yf_sym = row["yf_symbol"]
            fund = all_fundamentals.get(yf_sym, {})
            
            # Skip stocks with fundamental fetch errors
            if "error" in fund and not fund.get("price"):
                continue
            
            # Skip stocks with no price data
            if not fund.get("price") and not fund.get("currentPrice"):
                continue
            
            scores = compute_composite_score(fund)
            
            stock_entry = {
                "symbol": sym,
                "yf_symbol": yf_sym,
                "company_name": row["company_name"],
                "sector": (row.get("sector") if row.get("sector") not in ("Unknown", "", None) else None) or fund.get("sector") or "Unknown",
                "industry": (row.get("industry") if row.get("industry") not in ("Unknown", "", None) else None) or fund.get("industry") or "Unknown",
                "composite_score": round(scores["composite"], 2),
                "valuation_score": round(scores["valuation"], 2),
                "quality_score": round(scores["quality"], 2),
                "growth_score": round(scores["growth"], 2),
                "momentum_score": round(scores["momentum"], 2),
                # Key fundamentals for display
                "price": fund.get("price") or fund.get("currentPrice"),
                "market_cap": fund.get("marketCap"),
                "trailing_pe": fund.get("trailingPE"),
                "roe": fund.get("returnOnEquity"),
                "revenue_growth": fund.get("revenueGrowth"),
                "earnings_growth": fund.get("earningsGrowth"),
                "debt_to_equity": fund.get("debtToEquity"),
                "return_52w": fund.get("52WeekChange"),
                "return_6m": fund.get("6m_return_pct"),
                "analyst_recommendation": fund.get("recommendationKey"),
                "analyst_target_mean": fund.get("targetMeanPrice"),
                "num_analysts": fund.get("numberOfAnalystOpinions"),
            }
            scored_stocks.append(stock_entry)
        
        logger.info(f"Scored {len(scored_stocks)} stocks (excluded {len(universe_df) - len(scored_stocks)} with missing data)")
        
        # Step 4: Sort by composite score
        scored_stocks.sort(key=lambda x: x["composite_score"], reverse=True)

        # Step 4b: Blend FinBERT sentiment into composite score (top 150 only — API cost)
        if config.SENTIMENT_WEIGHT > 0:
            try:
                from data.sentiment import batch_sentiment_scores
                sentiment_candidates = scored_stocks[:150]
                sent_map = batch_sentiment_scores(
                    [{"symbol": s["symbol"], "company_name": s["company_name"]} for s in sentiment_candidates],
                    days=3,
                    skip_if_no_token=True,  # skip unless HF_TOKEN set; avoids DNS spam on restricted networks
                )
                w_sent = config.SENTIMENT_WEIGHT
                w_quant = 1.0 - w_sent
                blended = 0
                for s in sentiment_candidates:
                    raw = s["composite_score"]
                    sent = sent_map.get(s["symbol"])
                    if sent is None:
                        # Sentiment unavailable — keep quant score unchanged
                        s["sentiment_score"] = None
                        continue
                    # sentiment_score is -1..+1; rescale to 0..100 range contribution
                    sent_contribution = (sent + 1.0) * 50.0  # 0..100
                    s["sentiment_score"] = round(sent, 4)
                    s["composite_score"] = round(raw * w_quant + sent_contribution * w_sent, 2)
                    blended += 1
                scored_stocks.sort(key=lambda x: x["composite_score"], reverse=True)
                logger.info("Sentiment blended into composite (weight=%.0f%%, %d/%d stocks scored)", w_sent * 100, blended, len(sentiment_candidates))
            except Exception as e:
                logger.warning("Sentiment blend skipped: %s", e)

        # Add quant rank
        for i, stock in enumerate(scored_stocks):
            stock["quant_rank"] = i + 1

        # Get current macro context
        logger.info("Fetching macro context...")
        macro = get_macro_context()
        
        # Step 5: Optional LLM qualitative review of top 100 → selects top 50
        top100 = scored_stocks[:100]
        if not self.use_llm_review:
            logger.info("Stage 1: Skipping LLM qualitative review; using quant-ranked top %s.", config.STAGE1_TOP_N)
            top50 = self._quant_topn(top100, n=config.STAGE1_TOP_N)
            top50_enriched = self._enrich_top50(top50, scored_stocks)
            return self._save_and_return(
                start_time=start_time,
                universe_df=universe_df,
                scored_stocks=scored_stocks,
                top50_enriched=top50_enriched,
                save_path=save_path,
            )

        logger.info(f"Sending top 100 to LLM ({config.STAGE1_MODEL}) for qualitative review...")
        
        # Prepare compact summary for Claude
        top100_summary = []
        for s in top100:
            top100_summary.append({
                "rank": s["quant_rank"],
                "symbol": s["symbol"],
                "company_name": s["company_name"],
                "sector": s["sector"],
                "composite_score": s["composite_score"],
                "pe": s.get("trailing_pe"),
                "roe": s.get("roe"),
                "rev_growth": s.get("revenue_growth"),
                "earn_growth": s.get("earnings_growth"),
                "return_52w": s.get("return_52w"),
                "analyst_rec": s.get("analyst_recommendation"),
            })
        
        macro_summary = {
            "nifty50_1m": macro.get("nifty50_1m_return"),
            "nifty50_3m": macro.get("nifty50_3m_return"),
            "india_vix": macro.get("india_vix"),
            "usd_inr": macro.get("usd_inr"),
            "top_sectors_by_return": sorted(
                macro.get("sector_1m_returns", {}).items(),
                key=lambda x: x[1] if x[1] else -999,
                reverse=True,
            )[:5],
        }
        
        prompt = f"""Review and finalize the top 50 stocks for the Indian AI Portfolio pipeline.

Current macro environment:
{json.dumps(macro_summary, indent=2, default=str)}

Top 100 quantitatively-scored stocks (from Nifty 500):
{json.dumps(top100_summary, indent=2, default=str)}

Task:
1. Review the top 100 and select the best 50 to advance to adversarial research
2. Consider: sector balance, macro fit (which sectors benefit from current macro?), data quality
3. Flag any obvious exclusions (e.g., PSU banks with very high NPAs, promoter pledge issues)
4. You may reorder slightly based on qualitative factors — but stay close to the quant ranking
5. Return exactly 50 stocks in the specified JSON format"""
        
        claude_response = self.agent.run(prompt)

        # Parse LLM's top 50 (retry once with stricter JSON-only instruction)
        try:
            top50 = self._parse_top50(claude_response)
        except Exception as e:
            logger.warning("Stage 1 JSON parse failed (%s). Retrying once with stricter JSON-only instruction.", e)
            claude_response = self.agent.run(prompt + "\n\nReturn ONLY the JSON array. No markdown. No commentary.")
            top50 = self._parse_top50(claude_response)
        
        # Enrich with full fundamental data
        top50_enriched = self._enrich_top50(top50, scored_stocks)
        
        return self._save_and_return(
            start_time=start_time,
            universe_df=universe_df,
            scored_stocks=scored_stocks,
            top50_enriched=top50_enriched,
            save_path=save_path,
        )

    def _save_and_return(
        self,
        *,
        start_time: datetime,
        universe_df: pd.DataFrame,
        scored_stocks: list[dict],
        top50_enriched: list[dict],
        save_path: Optional[Path],
    ) -> list[dict]:
        duration = (datetime.now() - start_time).total_seconds()
        result = {
            "stage": 1,
            "run_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "universe_size": len(universe_df),
            "scored_stocks": len(scored_stocks),
            "top50": top50_enriched,
            "token_usage": self.agent.get_token_usage(),
        }

        if save_path:
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Stage 1 results saved to {save_path}")

        logger.info(f"✅ Stage 1 complete in {duration:.0f}s — {len(top50_enriched)} stocks advance")
        logger.info(f"   Token usage: {self.agent.get_token_usage()}")
        return top50_enriched

    def _quant_topn(self, scored: list[dict], n: int) -> list[dict]:
        topn = scored[:n]
        out = []
        for i, s in enumerate(topn, 1):
            out.append({
                "rank": i,
                "symbol": s.get("symbol"),
                "company_name": s.get("company_name"),
                "sector": s.get("sector"),
                "composite_score": s.get("composite_score"),
                "quant_rank": s.get("quant_rank", i),
                "analyst_rank": i,
                "advance_reason": f"Quant top {i} by composite score ({s.get('composite_score')})",
                "flags": [],
            })
        return out

    def _parse_top50(self, claude_response: str) -> list[dict]:
        """Parse Claude's top 50 selection from JSON response."""
        try:
            return extract_json(claude_response, expected=list)
        
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(f"Failed to parse Claude Stage 1 JSON: {e}")

    def _enrich_top50(self, top50: list[dict], all_scored: list[dict]) -> list[dict]:
        """Add full data from scored_stocks back to the top 50."""
        scored_map = {s["symbol"]: s for s in all_scored}
        
        enriched = []
        for item in top50:
            symbol = item.get("symbol", "")
            full_data = scored_map.get(symbol, {})
            
            # Merge: Claude's ranking on top of full fundamentals
            merged = {**full_data, **item}
            enriched.append(merged)
        
        return enriched


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    from pathlib import Path
    save_to = Path("/tmp/stage1_test.json")
    
    screener = Stage1Screener()
    top50 = screener.run(save_path=save_to)
    
    print(f"\n{'='*60}")
    print(f"TOP 50 STOCKS FOR ADVERSARIAL RESEARCH")
    print(f"{'='*60}")
    for stock in top50[:20]:
        print(f"{stock.get('rank', '?'):3}. {stock['symbol']:15} {stock['company_name'][:30]:30} "
              f"Score: {stock.get('composite_score', 0):.1f}  Sector: {stock['sector']}")
    print(f"\n... and {max(0, len(top50)-20)} more")
