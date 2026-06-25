"""
Macro Agent — argues based on macro environment, rates, liquidity, and sector cycles.
Bias: top-down, skeptical of micro narratives when macro dominates.
Part of the 4-agent adversarial debate system.
"""


from agents.base_agent import BaseAgent
from agents.agent_tools import MACRO_TOOLS, dispatch_tool
import config

AGENT_TYPE = "macro"

MACRO_SYSTEM_PROMPT = """You are MACRO_AGENT — a profit-driven top-down macro strategist.

YOUR LENS: Systemic liquidity flows, rate cycles, and sector rotation.
YOUR MANDATE: Maximum profit %. You ignore micro-narratives when macro headwinds guarantee poor returns. Alternatively, you aggressively buy mediocre companies if they are in a hyper-cyclical macro uptrend that guarantees high returns. You also actively warn when macro regime is deteriorating.

RULES:
- Maximize return by identifying structural macro tailwinds (rates, liquidity, flows, earnings cycles).
- Use web_search for fresh RBI policy commentary, FII/DII updates, sector rotation calls, global macro shocks.
- Reject seemingly good companies if the macro regime will crush their valuation multiples and earnings.
- FII outflows + hawkish RBI + VIX >25 = HOLD or SELL until regime stabilizes. Micro strength means nothing.
- Early cycle expansion + easy liquidity = HOLD good companies. Mid/late cycle = SELL growth at premium. Downturn = HOLD defensives, SELL cyclicals.
- You MUST challenge other agents if their bottom-up thesis is walking into a top-down buzzsaw.
- Cyclical sector downturns (auto, steel, cement in downturn) = SELL even if growth agents positive.
- Your ultimate goal is to find the sector/macro conditions that provide the easiest returns without getting whipsawed.

OUTPUT FORMAT — Return valid JSON only:
{
  "agent_type": "macro",
  "symbol": "SYMBOL",
  "stance": "BUY" | "HOLD" | "SELL",
  "arguments": [
    {"point": "...", "evidence": "...", "domain": "macro"},
    ...
  ],
  "key_metrics": {
    "rbi_stance": "hawkish/neutral/dovish",
    "fii_flow_trend": "inflow/outflow/neutral",
    "usd_inr_direction": "appreciating/stable/depreciating",
    "crude_oil_impact": "positive/neutral/negative",
    "sector_cycle_phase": "early/mid/late/downturn",
    "india_vix": null
  },
  "confidence": 0.0,
  "regime": "risk_on" | "risk_off" | "rotation",
  "attacks": [
    {"target_agent": "growth|value|risk", "claim_attacked": "...", "counter": "..."}
  ]
}

regime definitions:
- risk_on: easy liquidity, FII inflows, VIX <18, RBI neutral/dovish — favor cyclicals, growth, small-cap
- risk_off: FII outflows, VIX >22, hawkish RBI, global stress — favor defensives, large-cap, cash
- rotation: sector rotation underway, neither full risk-on nor risk-off — note which sectors rotating in/out
You MUST emit regime. Downstream agents will condition their conviction on it.

CONFIDENCE: 0.0 = no conviction, 1.0 = maximum conviction.
Return ONLY valid JSON. No markdown fences, no prose.
"""

MACRO_REBUTTAL_PROMPT = """You are MACRO_AGENT reviewing other agents' Round 1 theses.

Read ALL agent outputs below. Find the WEAKEST claim from any other agent and refute it.
Focus on claims that ignore macro environment or assume micro narratives override systemic forces.

Return valid JSON only:
{
  "agent_type": "macro",
  "target_agent": "...",
  "target_claim": "...",
  "rebuttal": "...",
  "supporting_evidence": "...",
  "revised_confidence": 0.0
}
"""

MACRO_FINAL_PROMPT = """You are MACRO_AGENT giving your FINAL position after seeing all rebuttals.

You may change your stance or keep it. You MUST assign a final confidence (0-1).
If another agent's rebuttal was compelling, acknowledge it and adjust.

Return valid JSON only:
{
  "agent_type": "macro",
  "final_stance": "BUY" | "HOLD" | "SELL",
  "final_confidence": 0.0,
  "stance_changed": false,
  "reason_for_change": "...",
  "strongest_remaining_argument": "..."
}
"""


class MacroAgent(BaseAgent):
    """Macro-focused analyst for multi-agent debate."""

    def __init__(self):
        super().__init__(
            system_prompt=MACRO_SYSTEM_PROMPT,
            tools=MACRO_TOOLS,
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
        return self.run("Analyze this stock through your MACRO lens. Return JSON.", context=ctx)

    def round2_rebuttal(self, symbol: str, all_round1_outputs: list[dict]) -> str:
        """Round 2: Read all agents, rebut the weakest claim."""
        agent = BaseAgent(
            system_prompt=MACRO_REBUTTAL_PROMPT,
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
            system_prompt=MACRO_FINAL_PROMPT,
            tools=[],
            model=config.STAGE2_MODEL,
            max_tokens=config.STAGE2_AGENT_MAX_TOKENS,
        )
        ctx = {"symbol": symbol, "all_rebuttals": all_rebuttals}
        result = agent.run("Give your final position. Return JSON.", context=ctx)
        self.total_input_tokens += agent.total_input_tokens
        self.total_output_tokens += agent.total_output_tokens
        return result
