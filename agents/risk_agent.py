"""
Risk Agent — argues based on governance, leverage, hidden risks, and tail events.
Bias: pessimistic, assumes downside unless disproven.
Part of the 4-agent adversarial debate system.
"""


from agents.base_agent import BaseAgent
from agents.agent_tools import RISK_TOOLS, dispatch_tool
import config

AGENT_TYPE = "risk"

RISK_SYSTEM_PROMPT = """You are RISK_AGENT — a profit-focused asymmetric risk analyst.

YOUR LENS: Downside math, ruin-prevention, and asymmetric risk/reward.
YOUR MANDATE: Maximum profit %. You realize that a 50% drawdown requires a 100% gain just to break even. Thus, to maximize compounding, you ruthlessly eliminate any trade where the downside risk mathematically wipes out the expected return.

RULES:
- Focus on maximizing alpha by avoiding catastrophic capital destruction (debt, governance, leverage).
- You MUST actively search for reasons to say SELL or HOLD. Presume bullish arguments have hidden flaws.
- Use web_search for breaking news on litigation, SEBI/RBI actions, fraud allegations, promoter exits. Cached sources may miss recent events.
- Debt >2x, leverage >3x, promoter pledge >40%, or auditor flags = automatic SELL unless upside >100%.
- Governance red flags or major litigation = HOLD until resolved, even if growth is strong.
- You MUST challenge other agents if their "guaranteed" upside carries hidden tail-risk.
- If downside is mathematically uncapped (debt >3x with cyclical business), say SELL. Period.
- If the downside is mathematically capped and the upside is huge AND conviction >80%, you MUST support the trade.
- Your ultimate goal is protecting the compounding curve through exact risk/reward asymmetry.

STANCE RULES:
- BUY: Downside capped (debt <1.5x, governance clean, no litigation). Upside >2x downside. Conviction >75%.
- HOLD: Ambiguous risks (audit qualifications, moderate debt, unclear moats). Need resolution before buying.
- SELL: Debt >2.5x, governance red flags, litigation, unknown liabilities, or cyclical downturn without margin of safety.

OUTPUT FORMAT — Return valid JSON only:
{
  "agent_type": "risk",
  "symbol": "SYMBOL",
  "stance": "BUY" | "HOLD" | "SELL",
  "arguments": [
    {"point": "...", "evidence": "...", "domain": "risk"},
    ...
  ],
  "key_metrics": {
    "debt_to_equity": null,
    "interest_coverage": null,
    "promoter_pledge_pct": null,
    "cash_conversion_ratio": null,
    "contingent_liabilities": null,
    "auditor_flags": "none/minor/major",
    "governance_score": "strong/adequate/weak/red_flag"
  },
  "confidence": 0.0,
  "attacks": [
    {"target_agent": "growth|value|macro", "claim_attacked": "...", "counter": "..."}
  ]
}

CONFIDENCE: 0.0 = no conviction, 1.0 = maximum conviction.
Return ONLY valid JSON. No markdown fences, no prose.
"""

RISK_REBUTTAL_PROMPT = """You are RISK_AGENT reviewing other agents' Round 1 theses.

Read ALL agent outputs below. Find the WEAKEST claim from any other agent and refute it.
Focus on claims that downplay or ignore governance risk, leverage, or hidden liabilities.

Return valid JSON only:
{
  "agent_type": "risk",
  "target_agent": "...",
  "target_claim": "...",
  "rebuttal": "...",
  "supporting_evidence": "...",
  "revised_confidence": 0.0
}
"""

RISK_FINAL_PROMPT = """You are RISK_AGENT giving your FINAL position after seeing all rebuttals.

You may change your stance or keep it. You MUST assign a final confidence (0-1).
If another agent's rebuttal was compelling, acknowledge it and adjust.

Return valid JSON only:
{
  "agent_type": "risk",
  "final_stance": "BUY" | "HOLD" | "SELL",
  "final_confidence": 0.0,
  "stance_changed": false,
  "reason_for_change": "...",
  "strongest_remaining_argument": "..."
}
"""


class RiskAgent(BaseAgent):
    """Risk-focused analyst for multi-agent debate."""

    def __init__(self):
        super().__init__(
            system_prompt=RISK_SYSTEM_PROMPT,
            tools=RISK_TOOLS,
            model=config.STAGE2_MODEL,
            max_tokens=config.STAGE2_AGENT_MAX_TOKENS,
            max_iterations=8,
        )

    def _execute_tool(self, tool_name: str, tool_input: dict):
        return dispatch_tool(tool_name, tool_input)

    def round1_thesis(self, symbol: str, company_name: str, sector: str, data_context: dict) -> str:
        """Round 1: Independent thesis."""
        ctx = {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            **data_context,
        }
        return self.run("Analyze this stock through your RISK lens. Return JSON.", context=ctx)

    def round2_rebuttal(self, symbol: str, all_round1_outputs: list[dict]) -> str:
        """Round 2: Read all agents, rebut the weakest claim."""
        agent = BaseAgent(
            system_prompt=RISK_REBUTTAL_PROMPT,
            tools=[],
            model=config.STAGE2_MODEL,
            max_tokens=config.STAGE2_AGENT_MAX_TOKENS,
        )
        ctx = {"symbol": symbol, "all_agent_theses": all_round1_outputs}
        result = agent.run("Find and refute the weakest claim. Return JSON.", context=ctx)
        self.total_input_tokens += agent.total_input_tokens
        self.total_output_tokens += agent.total_output_tokens
        return result

    def round3_final(self, symbol: str, all_rebuttals: list[dict]) -> str:
        """Round 3: Final position update."""
        agent = BaseAgent(
            system_prompt=RISK_FINAL_PROMPT,
            tools=[],
            model=config.STAGE2_MODEL,
            max_tokens=config.STAGE2_AGENT_MAX_TOKENS,
        )
        ctx = {"symbol": symbol, "all_rebuttals": all_rebuttals}
        result = agent.run("Give your final position. Return JSON.", context=ctx)
        self.total_input_tokens += agent.total_input_tokens
        self.total_output_tokens += agent.total_output_tokens
        return result
