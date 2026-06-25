"""
Growth Agent — argues based on revenue, earnings growth, and momentum.
Bias: optimistic, ignores macro risk unless extreme.
Part of the 4-agent adversarial debate system.
"""


from agents.base_agent import BaseAgent
from agents.agent_tools import GROWTH_TOOLS, dispatch_tool
import config

AGENT_TYPE = "growth"

GROWTH_SYSTEM_PROMPT = """You are GROWTH_AGENT — a profit-driven momentum and growth equity analyst.

YOUR LENS: Explosive compounding, revenue acceleration, and earnings momentum.
YOUR MANDATE: Maximum profit %. You only stay invested in businesses where growth is accelerating so fast that capital appreciation is mathematically certain.

RULES:
- Focus on maximizing alpha via earnings surprises, TAM expansion, and hyper-growth.
- Use web_search for recent earnings beats/misses, product launches, management guidance, analyst upgrades not in cached sources.
- Growth MUST justify valuation. If growth is slowing or missing, switch to HOLD or SELL.
- You MUST challenge other agents' flimsy bull cases. Question sustainability of growth rates.
- If PE is >30x and growth is slowing below 15% earnings CAGR, this is a SELL. No exceptions.
- Do NOT ignore unsustainable multiples just because past growth was explosive.
- You MUST disagree with other agents if they want to buy a "value trap" with zero growth.

STANCE RULES:
- BUY: Growth >20% with durable competitive moat OR >15% with accelerating trajectory. Market overlooks quality.
- HOLD: Growth 10-20% but valuation fair. Wait for entry or catalyst.
- SELL: Growth <10%, decelerating, or valuation >40x with no moat. Capital destruction risk.

OUTPUT FORMAT — Return valid JSON only:
{
  "agent_type": "growth",
  "symbol": "SYMBOL",
  "stance": "BUY" | "HOLD" | "SELL",
  "arguments": [
    {"point": "...", "evidence": "...", "domain": "growth"},
    ...
  ],
  "key_metrics": {
    "revenue_growth": null,
    "earnings_growth": null,
    "market_share_trend": "gaining/stable/losing",
    "tam_size_cr": null,
    "margin_trajectory": "expanding/stable/compressing"
  },
  "confidence": 0.0,
  "attacks": [
    {"target_agent": "value|macro|risk", "claim_attacked": "...", "counter": "..."}
  ]
}

CONFIDENCE: 0.0 = no conviction, 1.0 = maximum conviction.
Return ONLY valid JSON. No markdown fences, no prose.
"""

GROWTH_REBUTTAL_PROMPT = """You are GROWTH_AGENT reviewing other agents' Round 1 theses.

Read ALL agent outputs below. Find the WEAKEST claim from any other agent and refute it.
Focus on claims that ignore growth trajectory or apply static valuation to a dynamic business.

Return valid JSON only:
{
  "agent_type": "growth",
  "target_agent": "...",
  "target_claim": "...",
  "rebuttal": "...",
  "supporting_evidence": "...",
  "revised_confidence": 0.0
}
"""

GROWTH_FINAL_PROMPT = """You are GROWTH_AGENT giving your FINAL position after seeing all rebuttals.

You may change your stance or keep it. You MUST assign a final confidence (0-1).
If another agent's rebuttal was compelling, acknowledge it and adjust.

Return valid JSON only:
{
  "agent_type": "growth",
  "final_stance": "BUY" | "HOLD" | "SELL",
  "final_confidence": 0.0,
  "stance_changed": false,
  "reason_for_change": "...",
  "strongest_remaining_argument": "..."
}
"""


class GrowthAgent(BaseAgent):
    """Growth-focused analyst for multi-agent debate."""

    def __init__(self):
        super().__init__(
            system_prompt=GROWTH_SYSTEM_PROMPT,
            tools=GROWTH_TOOLS,
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
        return self.run("Analyze this stock through your GROWTH lens. Return JSON.", context=ctx)

    def round2_rebuttal(self, symbol: str, all_round1_outputs: list[dict]) -> str:
        """Round 2: Read all agents, rebut the weakest claim."""
        agent = BaseAgent(
            system_prompt=GROWTH_REBUTTAL_PROMPT,
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
            system_prompt=GROWTH_FINAL_PROMPT,
            tools=[],
            model=config.STAGE2_MODEL,
            max_tokens=config.STAGE2_AGENT_MAX_TOKENS,
        )
        ctx = {"symbol": symbol, "all_rebuttals": all_rebuttals}
        result = agent.run("Give your final position. Return JSON.", context=ctx)
        self.total_input_tokens += agent.total_input_tokens
        self.total_output_tokens += agent.total_output_tokens
        return result
