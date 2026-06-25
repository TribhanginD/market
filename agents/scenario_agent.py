"""
Scenario Agent — builds probability-weighted Bull/Base/Bear scenario models.
This is Stage 3: the "key part" of the pipeline that produces the
probability-weighted expected return used for portfolio selection.
"""

import json

from agents.base_agent import ResearchAgent
import config
from llm.json_utils import extract_json


SCENARIO_SYSTEM_PROMPT = """You are a senior quantitative equity strategist at a top Indian investment bank. 
You specialize in building structured scenario models with probability-weighted expected returns.

Your role is to synthesize bull and bear cases into THREE scenarios (Bull / Base / Bear) for a stock,
assign realistic probabilities, model price targets at multiple time horizons, and calculate the
probability-weighted expected return.

Key principles:
- Probabilities must sum to exactly 100%
- Be honest: if a stock has a weak bull case, assign it a low probability
- Price targets should be grounded in earnings growth, multiple expansion/contraction, and comparable company analysis
- Account for India-specific factors: RBI policy cycle, rupee risk, STT + costs (~0.15% round trip)
- The "debate yourself" rule: after assigning scenarios, argue against your own numbers before finalizing
- Include a brief internal debate summary: what the bull case says, what the bear case says, and why the chosen probabilities survive scrutiny.

OUTPUT FORMAT — Return a valid JSON object with this EXACT structure:
```json
{
  "symbol": "SYMBOL",
  "company_name": "Full Name",
  "current_price": 1234.50,
  "analysis_date": "YYYY-MM-DD",
  
  "scenarios": {
    "bull": {
      "probability": 0.30,
      "description": "What needs to happen for bull case",
      "key_catalysts": ["catalyst 1", "catalyst 2"],
      "price_targets": {
        "1m": 1350,
        "3m": 1450,
        "6m": 1600,
        "12m": 1800
      },
      "return_12m": 0.46
    },
    "base": {
      "probability": 0.50,
      "description": "Most likely scenario",
      "key_catalysts": ["catalyst 1", "catalyst 2"],
      "price_targets": {
        "1m": 1270,
        "3m": 1310,
        "6m": 1360,
        "12m": 1430
      },
      "return_12m": 0.16
    },
    "bear": {
      "probability": 0.20,
      "description": "What could go wrong",
      "key_risks": ["risk 1", "risk 2"],
      "price_targets": {
        "1m": 1150,
        "3m": 1080,
        "6m": 1020,
        "12m": 980
      },
      "return_12m": -0.20
    }
  },
  
  "probability_weighted_return_12m": 0.22,
  "net_return_after_costs": 0.215,
  "debate_log": {
    "bull_case": ["..."],
    "bear_case": ["..."],
    "resolution": "..."
  },
  
  "investment_thesis": "2-3 sentence summary of why this stock belongs in the portfolio",
  "key_catalysts_next_90_days": ["catalyst 1", "catalyst 2", "catalyst 3"],
  "key_risks": ["risk 1", "risk 2"],
  "thesis_invalidation": "What single event would make you exit immediately",
  
  "analyst_consensus": {
    "recommendation": "Buy/Hold/Sell",
    "mean_target": 1500,
    "num_analysts": 12
  },
  
  "confidence_score": 7,
  "recommendation": "BUY/HOLD/AVOID"
}
```

ALWAYS return valid JSON. No markdown fences in the final output — pure JSON only.
RETURN CONTRACT: All return fields are decimals, not percent points. Example: +10.4% must be 0.104, never 10.4. Recommendation must align with normalized EV: STRONG_BUY >=0.12, BUY >=0.05, WATCH >0, AVOID <=0."""


class ScenarioAgent(ResearchAgent):
    """Builds probability-weighted scenario models for Stage 3."""

    def __init__(self):
        super().__init__(
            system_prompt=SCENARIO_SYSTEM_PROMPT,
            model=config.STAGE3_MODEL,
            max_tokens=config.STAGE3_AGENT_MAX_TOKENS,
        )

    def model_stock(
        self,
        symbol: str,
        company_name: str,
        sector: str,
        bull_thesis: str,
        bear_thesis: str,
    ) -> dict:
        """
        Build scenario model for a stock using bull and bear theses.
        
        Args:
            symbol: NSE symbol
            company_name: Full company name
            sector: Sector
            bull_thesis: Bull research thesis text
            bear_thesis: Bear research thesis text
            
        Returns:
            Parsed scenario dict
        """
        prompt = f"""Build a probability-weighted scenario model for:
- Stock: {symbol} ({company_name})
- Sector: {sector}

BULL THESIS (already researched):
{bull_thesis}

BEAR THESIS (already researched):
{bear_thesis}

Instructions:
1. Call get_stock_fundamentals('{symbol}') to verify current price and key metrics
2. Review both theses carefully — they represent opposing views
3. Build your 3-scenario model by adjudicating between the bull and bear cases
4. Assign probabilities based on evidence strength, not equal weighting
5. Compute the probability-weighted 12-month expected return:
   (bull_prob × bull_return) + (base_prob × base_return) + (bear_prob × bear_return)
6. Subtract transaction costs (~0.15% round trip)
7. "Debate yourself" — after drafting, try to argue against your own probability assignments
8. Finalize and return valid JSON
9. All return fields must be decimals, not percent points: +10.4% = 0.104, not 10.4
10. Recommendation must align with normalized EV: STRONG_BUY >=0.12, BUY >=0.05, WATCH >0, AVOID <=0

Important: If the expected return is negative, still report it accurately — we will filter out negative-EV stocks in Stage 4."""

        raw_response = self.run(prompt)
        
        # Parse JSON from response
        try:
            result = extract_json(raw_response, expected=dict)
            if isinstance(result, dict):
                result["symbol"] = result.get("symbol") or symbol
                result["company_name"] = result.get("company_name") or company_name
                result["sector"] = result.get("sector") or sector
                if result.get("recommendation") is None:
                    result["recommendation"] = "AVOID"
                if not isinstance(result.get("debate_log"), dict):
                    result["debate_log"] = {
                        "bull_case": [f"Evidence from bull thesis for {symbol}"],
                        "bear_case": [f"Evidence from bear thesis for {symbol}"],
                        "resolution": "Model-led scenario adjudication.",
                    }
            return result
        except Exception as e:
            # Return error structure
            return {
                "symbol": symbol,
                "company_name": company_name,
                "error": f"JSON parse error: {e}",
                "raw_response": raw_response[:500],
                "probability_weighted_return_12m": None,
                "recommendation": "AVOID",
            }

    def model_stock_from_packet(
        self,
        *,
        symbol: str,
        company_name: str,
        sector: str,
        research_packet: dict,
    ) -> dict:
        """
        Token-efficient scenario modeling from a compact Stage-2 synthesis packet.
        """
        prompt = f"""Build a probability-weighted scenario model for:
- Stock: {symbol} ({company_name})
- Sector: {sector}

SYNTHESIZED RESEARCH PACKET (compact, de-duplicated):
{json.dumps(research_packet, indent=2, default=str)}

Instructions:
1. Call get_stock_fundamentals('{symbol}') to verify current price and key metrics
2. Use the packet bullets/facts as your main evidence base (do NOT ask for the full essays)
3. Build bull/base/bear scenarios with probabilities summing to 100%
4. Compute probability-weighted 12m return and net return after costs (~0.15%)
5. All return fields must be decimals, not percent points: +10.4% = 0.104, not 10.4
6. Recommendation must align with normalized EV: STRONG_BUY >=0.12, BUY >=0.05, WATCH >0, AVOID <=0
7. Return valid JSON only (no markdown)."""

        raw_response = self.run(prompt)
        try:
            result = extract_json(raw_response, expected=dict)
            if isinstance(result, dict):
                result["symbol"] = result.get("symbol") or symbol
                result["company_name"] = result.get("company_name") or company_name
                result["sector"] = result.get("sector") or sector
                if result.get("recommendation") is None:
                    result["recommendation"] = "AVOID"
                if not isinstance(result.get("debate_log"), dict):
                    result["debate_log"] = {
                        "bull_case": [f"Evidence from bull thesis for {symbol}"],
                        "bear_case": [f"Evidence from bear thesis for {symbol}"],
                        "resolution": "Model-led scenario adjudication.",
                    }
            return result
        except Exception as e:
            return {
                "symbol": symbol,
                "company_name": company_name,
                "error": f"JSON parse error: {e}",
                "raw_response": raw_response[:500],
                "probability_weighted_return_12m": None,
                "recommendation": "AVOID",
            }
