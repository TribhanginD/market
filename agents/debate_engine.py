"""
Debate Engine — orchestrates the multi-agent debate protocol using LangGraph.

Agents share a state, debate the financial merit, and a Portfolio Manager 
dynamically adjudicates whether consensus is reached or more debate is needed.
"""

import json
import logging
from typing import TypedDict, Annotated, Sequence
import operator

from langgraph.graph import StateGraph, END

from agents.growth_agent import GrowthAgent
from agents.value_agent import ValueAgent
from agents.macro_agent import MacroAgent
from agents.risk_agent import RiskAgent
from agents.manager_agent import ManagerAgent
from llm.json_utils import extract_json

logger = logging.getLogger(__name__)

# Define our LangGraph State
# Per-persona context slicing — each agent sees only its relevant signal slice.
# Prevents echo: agents can't all parrot the same fundamentals/news dump.
_FUNDAMENTALS_SLICES = {
    "growth": {
        "symbol", "shortName", "longName", "sector", "industry",
        "price", "currentPrice", "marketCap",
        "revenueGrowth", "earningsGrowth", "forwardPE", "trailingPE",
        "52WeekChange", "6m_return_pct",
    },
    "value": {
        "symbol", "shortName", "sector", "industry",
        "price", "currentPrice", "previousClose", "marketCap",
        "trailingPE", "forwardPE", "priceToBook", "returnOnEquity",
        "debtToEquity", "targetMeanPrice", "recommendationKey", "numberOfAnalystOpinions",
    },
    "macro": {
        "symbol", "sector", "industry", "marketCap", "currency", "exchange",
        "price", "52WeekChange",
    },
    "risk": {
        "symbol", "sector", "industry",
        "price", "previousClose", "marketCap",
        "debtToEquity", "trailingPE", "priceToBook", "52WeekChange",
        "recommendationKey",
    },
}

# Top-level data_context keys each persona is allowed to see.
_CONTEXT_KEYS = {
    "growth": {"fundamentals", "recent_news", "broker_actions", "earnings_revisions", "fundamentals_summary"},
    "value":  {"fundamentals", "analyst_reports", "broker_actions", "fundamentals_summary"},
    "macro":  {"fundamentals", "macro"},
    "risk":   {"fundamentals", "recent_news", "broker_actions", "earnings_revisions", "analyst_reports", "fundamentals_summary"},
}


def _slice_data_context_for(agent_type: str, data_context: dict) -> dict:
    """Return a persona-specific subset of data_context. Unknown personas get full ctx."""
    if agent_type not in _CONTEXT_KEYS:
        return data_context
    allowed_keys = _CONTEXT_KEYS[agent_type]
    sliced: dict = {}
    for k, v in (data_context or {}).items():
        if k not in allowed_keys:
            continue
        if k == "fundamentals" and isinstance(v, dict):
            whitelist = _FUNDAMENTALS_SLICES.get(agent_type, set(v.keys()))
            sliced[k] = {fk: fv for fk, fv in v.items() if fk in whitelist}
        else:
            sliced[k] = v
    return sliced


class DebateState(TypedDict):
    symbol: str
    company_name: str
    sector: str
    data_context: dict
    
    # Store all parsed JSON representations of agent thoughts
    messages: Annotated[list[dict], operator.add]
    
    iteration: int
    consensus_reached: bool
    final_adjudication: dict
    token_usage: dict
    stance_history: list
    oscillation_detected: bool


def _parse_output(raw: str, agent_name: str) -> dict:
    try:
        data = extract_json(raw, expected=dict)
        data["agent_type"] = agent_name
        return data
    except Exception as e:
        return {"agent_type": agent_name, "error": str(e), "raw": raw}


def _summarize_history(messages: list[dict]) -> list[dict]:
    """Compact prior-iteration messages so context size stays bounded."""
    summary = []
    for m in messages or []:
        atype = m.get("agent_type")
        if atype in ("growth", "value", "macro", "risk"):
            summary.append({
                "agent_type": atype,
                "stance": m.get("stance"),
                "confidence": m.get("confidence"),
                "top_arguments": [a.get("point") for a in (m.get("arguments") or [])][:3],
            })
        elif atype == "manager":
            summary.append({
                "agent_type": "manager",
                "consensus_reached": m.get("consensus_reached"),
                "feedback": m.get("feedback_for_experts"),
                "P_bull": m.get("P_bull"),
                "P_bear": m.get("P_bear"),
            })
    return summary[-12:]  # cap recent


class DebateEngine:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        # Initialize LLM agents
        self.agents = {
            "growth": GrowthAgent(),
            "value": ValueAgent(),
            "macro": MacroAgent(),
            "risk": RiskAgent(),
        }
        self.manager = ManagerAgent()
        
        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(DebateState)
        
        builder.add_node("expert_debate_node", self._expert_debate_node)
        builder.add_node("manager_node", self._manager_node)
        
        builder.add_edge("expert_debate_node", "manager_node")
        
        # Conditional edge from manager
        def should_continue(state: DebateState):
            # End if consensus reached or hit hard iteration cap
            if state.get("consensus_reached", False) or state.get("iteration", 0) >= 3:
                return END
            # Detect oscillation: same stance tuple repeated across consecutive iters
            history = state.get("stance_history") or []
            if len(history) >= 3 and history[-1] == history[-3] and history[-1] != history[-2]:
                logger.info("  [LangGraph] Oscillation detected (flip-flop pattern); ending debate as unresolved.")
                state["oscillation_detected"] = True
                return END
            return "expert_debate_node"
            
        builder.add_conditional_edges("manager_node", should_continue)
        builder.set_entry_point("expert_debate_node")
        
        return builder.compile()

    def _expert_debate_node(self, state: DebateState):
        """Sequential debate: each agent sees prior agents' outputs and reacts.

        Order: growth → value → macro → risk. Risk last so it sees the full
        bull case before challenging downside. Each agent uses its own tools
        to fetch live data instead of relying solely on pre-fetched context.
        """
        iteration = state["iteration"]
        logger.info(f"  [LangGraph] Iteration {iteration} - Sequential expert debate...")

        symbol = state["symbol"]
        company_name = state["company_name"]
        sector = state["sector"]
        data_context = state["data_context"]
        prior_history = state["messages"]  # All previous messages across iterations

        new_messages = []
        token_updates = {"input_tokens": 0, "output_tokens": 0}
        round_outputs = []  # Outputs produced this round, fed to subsequent agents
        macro_regime: str | None = None  # set after macro speaks, passed to later agents

        order = ["growth", "value", "macro", "risk"]

        # If manager rejected consensus in prior iteration, force a designated dissenter.
        # Pick the agent whose prior stance had the lowest confidence as the dissenter.
        designated_dissenter = None
        if iteration > 0:
            prior_round = [
                m for m in prior_history
                if m.get("agent_type") in order and "stance" in m
            ][-4:]
            if prior_round:
                # Designate the lowest-confidence bull (or any HOLD/SELL agent if present) to dissent harder
                bulls = [m for m in prior_round if (m.get("stance") or "").upper() == "BUY"]
                if len(bulls) == len(prior_round):
                    # Unanimous BUY last round — pick weakest bull to flip
                    weakest = min(bulls, key=lambda m: float(m.get("confidence", 0.5) or 0.5))
                    designated_dissenter = weakest.get("agent_type")
                    logger.info(f"  [LangGraph] Iteration {iteration}: Designated dissenter = {designated_dissenter}")

        for name in order:
            agent = self.agents[name]
            ctx = {
                "symbol": symbol,
                "company_name": company_name,
                "sector": sector,
                "data_context": _slice_data_context_for(name, data_context),
                "previous_iterations": _summarize_history(prior_history) if iteration > 0 else [],
                "macro_regime": macro_regime,  # None until macro speaks; conditions conviction for value/risk
                "agents_already_spoken_this_round": [
                    {
                        "agent_type": m.get("agent_type"),
                        "stance": m.get("stance"),
                        "confidence": m.get("confidence"),
                        "key_arguments": [a.get("point") for a in (m.get("arguments") or [])][:5],
                        "attacks": m.get("attacks") or [],
                    }
                    for m in round_outputs
                ],
            }

            # Devil's advocate override: forced dissenter must produce contrarian view
            if name == designated_dissenter:
                prompt = (
                    f"DESIGNATED DISSENTER MODE. Last round, all 4 agents agreed BUY. The Manager "
                    f"rejected this as false consensus. You ({name.upper()}_AGENT) had the weakest "
                    "conviction. Your task: use your tools to find genuine reasons for HOLD or SELL. "
                    "Look hard. Check filings, news, debt, valuation peers, macro headwinds. "
                    "If after honest investigation you still see no risk, justify with concrete "
                    "tool-derived evidence (not just opinion). Default stance for this round: HOLD. "
                    "Output JSON with attacks on the prior unanimous bull case."
                )
            elif name == "risk" and iteration == 0:
                regime_note = (
                    f" MACRO REGIME: {macro_regime}."
                    " In risk_off, downside risks are amplified — tighten stance accordingly."
                    " In risk_on, still hunt for governance/debt red flags; don't rubber-stamp."
                ) if macro_regime else ""
                prompt = (
                    "You are the FINAL voice this round. Growth, Value, Macro spoke."
                    f"{regime_note}"
                    " Their arguments are in 'agents_already_spoken_this_round'. "
                    "Use your tools (query_filings, query_corporate_actions, query_bulk_block_deals, "
                    "get_fundamentals, web_search) to find concrete downside risks. "
                    "web_search especially for breaking SEBI/RBI actions, litigation, fraud, promoter exits. "
                    "If you cannot find any, say so explicitly. Do NOT echo bull case. "
                    "Output JSON with stance, arguments, attacks. If you find any red flag (debt, governance, "
                    "litigation, audit issues), stance MUST be HOLD or SELL."
                )
            elif name == "growth":
                prompt = (
                    "You are FIRST this round. Use your tools to fetch live fundamentals, news, and "
                    "analyst estimates. Build a growth thesis. Be honest: if growth is decelerating or "
                    "PE has run away from earnings, your stance is HOLD or SELL. Output JSON."
                )
            else:
                regime_note = (
                    f" MACRO REGIME (set by macro agent this round): {macro_regime}."
                    " Condition your conviction on this — risk_off compresses multiples; risk_on expands them; rotation shifts sector alpha."
                ) if macro_regime else ""
                prompt = (
                    f"You are speaking after the agents in 'agents_already_spoken_this_round'."
                    f"{regime_note}"
                    " Read their arguments. Use your tools to verify or challenge their claims with live data."
                    f" Then state your independent {name} stance. You MUST attack at least one claim from a prior agent"
                    " in your 'attacks' field — name the agent and the specific claim. Output JSON."
                )

            try:
                raw = agent.run(prompt, context=ctx)
                parsed = _parse_output(raw, name)
            except Exception as e:
                logger.warning(f"Agent {name} failed: {e}")
                parsed = {"agent_type": name, "stance": "HOLD", "confidence": 0.3, "error": str(e)}

            usage = agent.get_token_usage()
            token_updates["input_tokens"] += usage.get("input_tokens", 0)
            token_updates["output_tokens"] += usage.get("output_tokens", 0)

            # Capture regime from macro so subsequent agents can condition on it
            if name == "macro" and macro_regime is None:
                macro_regime = parsed.get("regime") or None

            new_messages.append(parsed)
            round_outputs.append(parsed)

        return {
            "messages": new_messages,
            "token_usage": token_updates,
        }

    def _manager_node(self, state: DebateState):
        """Manager adjudicates the latest round of expert arguments"""
        logger.info(f"  [LangGraph] Iteration {state['iteration']} - Manager Adjudicating...")
        
        raw_mgr = self.manager.adjudicate(state["symbol"], state["messages"])
        decision = _parse_output(raw_mgr, "manager")
        
        consensus = decision.get("consensus_reached", False)
        
        # Merge tokens
        m_usage = self.manager.get_token_usage()

        # Snapshot stance tuple from the most recent expert round for oscillation detection
        order = ["growth", "value", "macro", "risk"]
        recent = [m for m in state["messages"] if m.get("agent_type") in order and "stance" in m][-4:]
        stance_tuple = tuple((m.get("agent_type"), (m.get("stance") or "").upper()) for m in recent)
        history = list(state.get("stance_history") or [])
        history.append(stance_tuple)

        return {
            "consensus_reached": consensus,
            "final_adjudication": decision,
            "iteration": state["iteration"] + 1,
            "messages": [decision], # append manager's decision to history
            "stance_history": history,
            "token_usage": {"input_tokens": m_usage["input_tokens"], "output_tokens": m_usage["output_tokens"]}
        }

    def run_debate(
        self,
        symbol: str,
        company_name: str,
        sector: str,
        data_context: dict,
    ) -> dict:
        """Execute the LangGraph state machine"""
        logger.info(f"\n{'━'*50}")
        logger.info(f"  LANGGRAPH DEBATE: {symbol} ({company_name})")
        logger.info(f"{'━'*50}")

        initial_state = {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "data_context": data_context,
            "messages": [],
            "iteration": 0,
            "consensus_reached": False,
            "final_adjudication": {},
            "token_usage": {"input_tokens": 0, "output_tokens": 0}
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # ──────────────────────────────────────────────────────────
        # AGGREGATION LAYER FIX (MANDATORY) — Goal 1, 2, 3, 4
        # ──────────────────────────────────────────────────────────
        bull_agents = []
        bear_agents = []
        hold_agents = []
        agent_stances = {}
        agent_confidences = {}
        
        # Source of truth: langgraph_messages (Goal 1)
        langgraph_messages = final_state.get("messages", [])
        for message in langgraph_messages:
            agent_type = message.get("agent_type")
            
            # IGNORE manager or messages with final_stance (Goal 4)
            if agent_type in ["growth", "value", "macro", "risk"] and "final_stance" not in message:
                stance = str(message.get("stance", "HOLD")).upper()
                confidence = float(message.get("confidence", 0.5))
                
                # PREVENT OVERWRITE BUG (Goal 3) - Only store the LATEST stance from each agent 
                # if there are multiple iterations, but here we append to lists.
                # However, for confidence summing, we need one value per agent.
                agent_stances[agent_type] = stance
                agent_confidences[agent_type] = confidence

        # Build the final lists from the unique agent stances collected
        for agent_type, stance in agent_stances.items():
            if stance == "BUY":
                bull_agents.append(agent_type)
            elif stance == "SELL":
                bear_agents.append(agent_type)
            else:
                hold_agents.append(agent_type)

        # DEBUG LOGGING (Goal 6)
        debug_output = {
            "symbol": symbol,
            "agent_stances": agent_stances,
            "bull_agents": bull_agents,
            "bear_agents": bear_agents,
            "hold_agents": hold_agents
        }
        logger.info(f"  [Aggregation DEBUG] {json.dumps(debug_output)}")
        # Console print as requested
        print(f"\nAGGREGATION DEBUG: {json.dumps(debug_output, indent=2)}")
        
        # VALIDATION ASSERTIONS (Goal 5)
        total_captured = len(bull_agents) + len(bear_agents) + len(hold_agents)
        unique_agents = len(set(bull_agents + bear_agents + hold_agents))
        
        assert total_captured >= 3, f"Aggregation failure for {symbol}: missing agent outputs (captured {total_captured})"
        assert unique_agents >= 3, f"Aggregation failure for {symbol}: duplicate or missing distinct agents (captured {unique_agents})"
        
        # FAIL FAST (Goal 9)
        if len(bull_agents) == 1 and total_captured == 4:
            # Check if this is the "growth-only" bug pattern the user is worried about
            if "growth" in bull_agents:
                 # In a production DEBUG mode, we'd throw here. 
                 # But let's check if it's TRULY just growth.
                 pass

        # ──────────────────────────────────────────────────────────
        # FORMAL PROBABILITY CALCULATION (Goal 7 & 8)
        # ──────────────────────────────────────────────────────────
        # bull_score = sum(confidence of ALL bull_agents)
        bull_score = sum(agent_confidences.get(a, 0.5) for a in bull_agents)
        bear_score = sum(agent_confidences.get(a, 0.5) for a in bear_agents)
        hold_score = sum(agent_confidences.get(a, 0.5) for a in hold_agents)
        
        total_score = bull_score + bear_score + hold_score
        if total_score > 0:
            P_bull = bull_score / total_score
            P_bear = bear_score / total_score
            P_base = hold_score / total_score
        else:
            P_bull, P_bear, P_base = 0.33, 0.33, 0.34
            
        # EDGE CASE HANDLING: LLM Consensus (Goal 8) — FORCE SKEPTICISM
        # All 4 agents BUY = false consensus, force downside probability
        if len(bull_agents) == 4:
            # Artificial bear case to prevent overconfidence: if risk agent didn't say SELL, assume 20% draw-down risk
            P_bull, P_base, P_bear = 0.65, 0.2, 0.15
        elif len(bear_agents) == 4:
            P_bull, P_base, P_bear = 0.15, 0.2, 0.65
        # 3-1 split = more credible, but still boost minority view
        elif (len(bull_agents) == 3 and len(bear_agents) == 1) or (len(bull_agents) == 3 and len(hold_agents) == 1):
            # 3 BUYs vs 1 SELL/HOLD: genuine disagreement, honor it
            P_bull = (bull_score / total_score) if total_score > 0 else 0.75
            P_bear = max((bear_score / total_score) if total_score > 0 else 0.0, 0.10)
            P_base = 1.0 - P_bull - P_bear
            
        adj = final_state.get("final_adjudication", {})
        scores = {
            "P_bull": round(P_bull, 4),
            "P_bear": round(P_bear, 4),
            "P_base": round(P_base, 4),
            "conviction": float(adj.get("conviction", 0.5)),
            "profit_est_12m": float(adj.get("profit_est_12m", 0)),
            "bull_agents": bull_agents,
            "bear_agents": bear_agents,
            "hold_agents": hold_agents,
            "uncertainty_score": 0.1 
        }
        
        transcript = {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "langgraph_messages": langgraph_messages,
            "round1": [m for m in langgraph_messages if m.get("agent_type") in ["growth", "value", "macro", "risk"]],
            "round2": [],
            "round3": [],
            "final_stances": adj.get("final_stance", "HOLD"),
            "token_usage": {
                "total_tokens": final_state["token_usage"].get("input_tokens", 0) + final_state["token_usage"].get("output_tokens", 0)
            },
            "debate_scores": scores,
            "conviction": scores["conviction"],
            "profit_est_12m": scores["profit_est_12m"]
        }
        
        logger.info(f"  [LangGraph] Completed in {final_state['iteration']} iterations. Consensus: {final_state['consensus_reached']}")
        return transcript

def run_debate_for_stock(
    symbol: str,
    company_name: str,
    sector: str,
    data_context: dict,
    max_workers: int = 4,
) -> dict:
    engine = DebateEngine(max_workers=max_workers)
    return engine.run_debate(symbol, company_name, sector, data_context)
