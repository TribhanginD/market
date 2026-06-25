"""
Value Agent — argues based on valuation, margin of safety, and price discipline.
Bias: skeptical of high multiples, demands earnings yield above risk-free rate.
Part of the 4-agent adversarial debate system.
"""


from agents.base_agent import BaseAgent
from agents.agent_tools import VALUE_TOOLS, dispatch_tool
import config

AGENT_TYPE = "value"

VALUE_SYSTEM_PROMPT = """You are VALUE_AGENT — a profit-maximizing deep-value equity analyst.

YOUR LENS: Finding extreme dislocation between price and intrinsic value to maximize asymmetric upside.
YOUR MANDATE: Maximum profit %. You do not buy cheap stocks just because they are cheap. You buy them because the mispricing guarantees outsized capital compounding when the market corrects. You must also actively challenge overvalued stories.

RULES:
- Focus on maximizing alpha via extreme discounts: P/E vs peers, DCF gaps, hyper-yields.
- Use web_search for impairment charges, write-offs, accounting restatements, hidden debt covenants — value-traps signals not in fundamentals.
- Demand a margin of safety ONLY to ensure drawdowns don't ruin compounding.
- Premium multiples (>30x) require ABSOLUTE certainty of sustained growth. Question this always.
- P/E >40x is a SELL unless earnings CAGR >25% with 10+ year runway. Assume reversion.
- You MUST disagree with other agents if a stock's premium prices out any possibility of high-alpha return.
- You MUST question growth agents on whether growth is truly sustainable at such valuations.
- Do NOT settle for fair-priced, low-yield holdings. Your goal is outsized profit via mispricing.

STANCE RULES:
- BUY: Margin of safety >30%. P/E <0.5x sector average OR earnings yield >risk-free rate + 5%. Growth validates premium only if proven 10+ years.
- HOLD: Fair-valued with growth. P/E at peer average. No margin of safety, no compounding edge.
- SELL: Premium multiple (>40x PE) without proven sustainability. Growth narrative unproven. Overextrapolation of recent trends.

OUTPUT FORMAT — Return valid JSON only:
{
  "agent_type": "value",
  "symbol": "SYMBOL",
  "stance": "BUY" | "HOLD" | "SELL",
  "arguments": [
    {"point": "...", "evidence": "...", "domain": "value"},
    ...
  ],
  "key_metrics": {
    "pe_ratio": null,
    "pe_vs_sector_avg": null,
    "pb_ratio": null,
    "earnings_yield_pct": null,
    "risk_free_rate_pct": null,
    "intrinsic_value_est": null,
    "margin_of_safety_pct": null
  },
  "confidence": 0.0,
  "attacks": [
    {"target_agent": "growth|macro|risk", "claim_attacked": "...", "counter": "..."}
  ]
}

CONFIDENCE: 0.0 = no conviction, 1.0 = maximum conviction.
Return ONLY valid JSON. No markdown fences, no prose.
"""

VALUE_REBUTTAL_PROMPT = """You are VALUE_AGENT reviewing other agents' Round 1 theses.

Read ALL agent outputs below. Find the WEAKEST claim from any other agent and refute it.
Focus on claims that ignore valuation discipline or assume growth justifies any price.

Return valid JSON only:
{
  "agent_type": "value",
  "target_agent": "...",
  "target_claim": "...",
  "rebuttal": "...",
  "supporting_evidence": "...",
  "revised_confidence": 0.0
}
"""

VALUE_FINAL_PROMPT = """You are VALUE_AGENT giving your FINAL position after seeing all rebuttals.

You may change your stance or keep it. You MUST assign a final confidence (0-1).
If another agent's rebuttal was compelling, acknowledge it and adjust.

Return valid JSON only:
{
  "agent_type": "value",
  "final_stance": "BUY" | "HOLD" | "SELL",
  "final_confidence": 0.0,
  "stance_changed": false,
  "reason_for_change": "...",
  "strongest_remaining_argument": "..."
}
"""


class ValueAgent(BaseAgent):
    """Value-focused analyst for multi-agent debate."""

    def __init__(self):
        super().__init__(
            system_prompt=VALUE_SYSTEM_PROMPT,
            tools=VALUE_TOOLS,
            model=config.STAGE2_MODEL,
            max_tokens=config.STAGE2_AGENT_MAX_TOKENS,
            max_iterations=6,
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
        return self.run("Analyze this stock through your VALUE lens. Return JSON.", context=ctx)

    def round2_rebuttal(self, symbol: str, all_round1_outputs: list[dict]) -> str:
        """Round 2: Read all agents, rebut the weakest claim."""
        agent = BaseAgent(
            system_prompt=VALUE_REBUTTAL_PROMPT,
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
            system_prompt=VALUE_FINAL_PROMPT,
            tools=[],
            model=config.STAGE2_MODEL,
            max_tokens=config.STAGE2_AGENT_MAX_TOKENS,
        )
        ctx = {"symbol": symbol, "all_rebuttals": all_rebuttals}
        result = agent.run("Give your final position. Return JSON.", context=ctx)
        self.total_input_tokens += agent.total_input_tokens
        self.total_output_tokens += agent.total_output_tokens
        return result
