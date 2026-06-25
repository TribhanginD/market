"""
AG2 GroupChat-based Stage 2 debate engine.

Drop-in replacement for `agents.debate_engine.DebateEngine` using AutoGen/AG2
ConversableAgent + GroupChat for natural multi-turn debate.

Same external interface as DebateEngine:
    engine = AG2DebateEngine()
    transcript = engine.run_debate(symbol, company_name, sector, data_context)

Output transcript matches the legacy schema so Stage 3 / persistence layers
need no changes.
"""

import json
import logging
import os
import re


import config
from agents.agent_tools import (
    GROWTH_TOOLS, VALUE_TOOLS, MACRO_TOOLS, RISK_TOOLS,
    dispatch_tool,
)
from agents.growth_agent import GROWTH_SYSTEM_PROMPT
from agents.value_agent import VALUE_SYSTEM_PROMPT
from agents.macro_agent import MACRO_SYSTEM_PROMPT
from agents.risk_agent import RISK_SYSTEM_PROMPT
from agents.manager_agent import MANAGER_SYSTEM_PROMPT
from llm.json_utils import extract_json

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# LLM Config
# ──────────────────────────────────────────────────────────

def _build_llm_config(model_override: str = None) -> dict:
    """Build AG2 llm_config from project env. Prefer OpenAI-compat, fallback Anthropic."""
    base_url = config.OPENAI_COMPAT_BASE_URL
    api_key = config.OPENAI_COMPAT_API_KEY
    model = model_override or config.OPENAI_COMPAT_MODEL or config.STAGE2_MODEL

    # Strip provider prefix (e.g. "openai:Qwen/...")
    if ":" in model:
        model = model.split(":", 1)[1]

    if base_url and api_key:
        config_list = [{
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "api_type": "openai",
            "price": [0.0, 0.0],  # silence cost warnings
        }]
    elif config.ANTHROPIC_API_KEY:
        config_list = [{
            "model": model if model.startswith("claude") else "claude-3-5-sonnet-20241022",
            "api_key": config.ANTHROPIC_API_KEY,
            "api_type": "anthropic",
        }]
    else:
        raise RuntimeError("AG2 debate needs either OPENAI_COMPAT_* or ANTHROPIC_API_KEY")

    return {
        "config_list": config_list,
        "cache_seed": None,  # disable cache; per-stock context differs
        "temperature": 0.4,
        "timeout": 120,
    }


# ──────────────────────────────────────────────────────────
# Tool function builders (AG2 needs python callables, not just schemas)
# ──────────────────────────────────────────────────────────

def _make_tool_callable(tool_name: str):
    """Wrap dispatch_tool as a typed python function for AG2 registration."""
    def _call(**kwargs):
        result = dispatch_tool(tool_name, kwargs)
        # Compact for chat injection
        try:
            return json.dumps(result, default=str)[:6000]
        except Exception:
            return str(result)[:6000]
    _call.__name__ = tool_name
    _call.__doc__ = f"Call tool '{tool_name}' with kwargs matching its schema."
    return _call


_TOOL_DESCRIPTIONS = {
    "get_fundamentals": "Fetch stock fundamentals (PE, ROE, growth, debt, technicals). Args: symbol",
    "get_recent_news": "Recent stock news. Args: symbol, company_name, days",
    "get_macro_context": "Indian macro: Nifty, USD/INR, VIX, sector returns. No args",
    "get_fii_dii_flows": "FII/DII flows in INR cr. No args",
    "get_sector_peer_valuations": "Peer PE/PB/ROE distribution. Args: symbol, sector",
    "query_filings": "NSE/BSE filings for stock. Args: symbol, days (default 90)",
    "query_corporate_actions": "Dividends/splits/buybacks. Args: symbol, days (default 180)",
    "query_bulk_block_deals": "Bulk + block deals. Args: symbol, days (default 60)",
    "get_upcoming_results": "Upcoming earnings dates. Args: symbol",
    "get_analyst_reports": "Analyst reports. Args: symbol, company_name, days",
    "get_symbol_memory": "Prior debate memory + ETL summary. Args: symbol",
    "web_search": "Live web search via Tavily. Args: query, search_depth(basic|advanced), max_results, days",
}


def _register_tools_for_agent(agent, executor, tool_schemas: list[dict]) -> None:
    """Register each tool from a schema list onto the AG2 agent."""
    from autogen import register_function

    for schema in tool_schemas:
        name = schema["name"]
        desc = _TOOL_DESCRIPTIONS.get(name, schema.get("description", ""))[:200]
        fn = _make_tool_callable(name)
        try:
            register_function(
                fn,
                caller=agent,
                executor=executor,
                name=name,
                description=desc,
            )
        except Exception as e:
            logger.warning(f"Tool register failed {name}: {e}")


# ──────────────────────────────────────────────────────────
# AG2 Debate Engine
# ──────────────────────────────────────────────────────────

_AGENT_ORDER = ["growth", "value", "macro", "risk", "manager"]

_PROMPT_SUFFIX = (
    "\n\nDEBATE PROTOCOL:\n"
    "- Read the entire chat history before speaking.\n"
    "- Use your tools to verify peer claims with live data.\n"
    "- Attack at least one weak claim from a prior speaker by name.\n"
    "- End your turn with a fenced JSON block containing fields: "
    '`stance` (BUY/HOLD/SELL), `confidence` (0-1), `arguments` (list), `attacks` (list).\n'
    "- Do not repeat prior consensus. Add new evidence or dissent.\n"
)

_MANAGER_SUFFIX = (
    "\n\nMANAGER PROTOCOL:\n"
    "- Read the round of expert outputs.\n"
    "- Reject false consensus: if all 4 BUY without explicit downside review, "
    "set `consensus_reached: false` and demand more debate.\n"
    "- End turn with a fenced JSON block: `consensus_reached`, `final_stance`, "
    "`P_bull`, `P_base`, `P_bear`, `conviction`, `profit_est_12m`, `feedback_for_experts`.\n"
    "- If `consensus_reached: true`, end your message with the literal token TERMINATE.\n"
)


class AG2DebateEngine:
    def __init__(self, max_workers: int = 4, max_rounds: int = 3):
        self.max_workers = max_workers
        self.max_rounds = max_rounds
        self.llm_config = _build_llm_config()

    def _build_chat(self, symbol: str, company_name: str, sector: str, data_context: dict):
        from autogen import ConversableAgent, GroupChat, GroupChatManager

        ctx_blob = f"""
Symbol: {symbol} | Company: {company_name} | Sector: {sector}

PRE-FETCHED DATA CONTEXT (use tools for deeper queries):
{json.dumps(_compact_context(data_context), default=str)[:3500]}
""".strip()

        growth = ConversableAgent(
            name="growth",
            system_message=GROWTH_SYSTEM_PROMPT + _PROMPT_SUFFIX,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
        value = ConversableAgent(
            name="value",
            system_message=VALUE_SYSTEM_PROMPT + _PROMPT_SUFFIX,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
        macro = ConversableAgent(
            name="macro",
            system_message=MACRO_SYSTEM_PROMPT + _PROMPT_SUFFIX,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
        risk = ConversableAgent(
            name="risk",
            system_message=RISK_SYSTEM_PROMPT + _PROMPT_SUFFIX,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
        manager = ConversableAgent(
            name="manager",
            system_message=MANAGER_SYSTEM_PROMPT + _MANAGER_SUFFIX,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda m: isinstance(m, dict) and "TERMINATE" in (m.get("content") or ""),
        )

        # Tool executor: a silent agent that runs tool calls
        executor = ConversableAgent(
            name="tool_executor",
            system_message="Silent tool executor.",
            llm_config=False,  # no LLM
            human_input_mode="NEVER",
            default_auto_reply="",
        )

        _register_tools_for_agent(growth, executor, GROWTH_TOOLS)
        _register_tools_for_agent(value, executor, VALUE_TOOLS)
        _register_tools_for_agent(macro, executor, MACRO_TOOLS)
        _register_tools_for_agent(risk, executor, RISK_TOOLS)

        agents_in_order = [growth, value, macro, risk, manager]

        def speaker_selection(last_speaker, groupchat):
            """Round-robin growth → value → macro → risk → manager → growth ..."""
            # Skip executor turns (tool execution doesn't change debate position)
            if last_speaker.name == "tool_executor":
                return last_speaker  # let executor finish its tool reply
            try:
                idx = _AGENT_ORDER.index(last_speaker.name)
                next_idx = (idx + 1) % len(_AGENT_ORDER)
                return agents_in_order[next_idx]
            except ValueError:
                return growth  # fallback start

        groupchat = GroupChat(
            agents=agents_in_order + [executor],
            messages=[],
            max_round=self.max_rounds * len(_AGENT_ORDER) + 8,  # 8 buffer for tool turns
            speaker_selection_method=speaker_selection,
            allow_repeat_speaker=True,
        )

        chat_manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config,
            is_termination_msg=lambda m: isinstance(m, dict) and "TERMINATE" in (m.get("content") or ""),
        )

        return chat_manager, groupchat, growth, ctx_blob

    def run_debate(self, symbol: str, company_name: str, sector: str, data_context: dict) -> dict:
        logger.info(f"\n{'━'*50}\n  AG2 DEBATE: {symbol} ({company_name})\n{'━'*50}")

        chat_manager, groupchat, growth, ctx_blob = self._build_chat(
            symbol, company_name, sector, data_context
        )

        opener = (
            f"Stock under debate. {ctx_blob}\n\n"
            "growth_agent speaks first. Build initial thesis with tool calls. "
            "Then value, macro, risk respond in order. Manager adjudicates. "
            "Goal: reach consensus on stance, with genuine bull/bear argument capture."
        )

        try:
            growth.initiate_chat(
                recipient=chat_manager,
                message=opener,
                clear_history=True,
            )
        except Exception as e:
            logger.warning(f"AG2 chat failed for {symbol}: {e}")

        return _build_transcript(symbol, company_name, sector, groupchat.messages)


# ──────────────────────────────────────────────────────────
# Transcript builder — match legacy DebateEngine schema
# ──────────────────────────────────────────────────────────

def _compact_context(data_context: dict) -> dict:
    keep = {}
    for k in ("fundamentals", "macro", "fundamentals_summary"):
        if k in data_context:
            keep[k] = data_context[k]
    if "recent_news" in data_context:
        keep["recent_news"] = data_context["recent_news"][:5]
    return keep


def _extract_last_json(content: str) -> dict:
    """Pull the LAST fenced JSON block from a chat message."""
    if not content:
        return {}
    # find fenced blocks
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if not blocks:
        # try bare JSON
        try:
            return extract_json(content, expected=dict) or {}
        except Exception:
            return {}
    for blob in reversed(blocks):
        try:
            return json.loads(blob)
        except Exception:
            continue
    return {}


def _build_transcript(symbol: str, company_name: str, sector: str, messages: list) -> dict:
    """Convert AG2 chat messages into legacy transcript format expected by Stage 3."""
    parsed_per_agent = {}  # agent_type -> latest parsed JSON
    langgraph_messages = []

    for msg in messages or []:
        name = (msg.get("name") or "").lower()
        if name not in ("growth", "value", "macro", "risk", "manager"):
            continue
        content = msg.get("content") or ""
        parsed = _extract_last_json(content)
        if not parsed:
            continue
        parsed["agent_type"] = name
        # Keep latest output per agent
        parsed_per_agent[name] = parsed
        langgraph_messages.append(parsed)

    # Manager adjudication = last manager message
    mgr = parsed_per_agent.get("manager", {})

    # Aggregate stances
    bull_agents, bear_agents, hold_agents = [], [], []
    confidences = {}
    for atype in ("growth", "value", "macro", "risk"):
        out = parsed_per_agent.get(atype, {})
        stance = (out.get("stance") or "HOLD").upper()
        conf = float(out.get("confidence") or 0.5)
        confidences[atype] = conf
        if stance == "BUY":
            bull_agents.append(atype)
        elif stance == "SELL":
            bear_agents.append(atype)
        else:
            hold_agents.append(atype)

    # Probabilities: prefer manager's, otherwise compute from confidences
    P_bull = mgr.get("P_bull")
    P_base = mgr.get("P_base")
    P_bear = mgr.get("P_bear")
    if P_bull is None or P_bear is None or P_base is None:
        bull_score = sum(confidences[a] for a in bull_agents)
        bear_score = sum(confidences[a] for a in bear_agents)
        hold_score = sum(confidences[a] for a in hold_agents)
        total = bull_score + bear_score + hold_score
        if total > 0:
            P_bull = bull_score / total
            P_bear = bear_score / total
            P_base = hold_score / total
        else:
            P_bull, P_base, P_bear = 0.34, 0.33, 0.33

    # Floor for P_bear when unanimous BUY (anti-false-consensus)
    if len(bull_agents) == 4 and (P_bear or 0) < 0.15:
        P_bull, P_base, P_bear = 0.65, 0.20, 0.15

    conviction = float(mgr.get("conviction") or max(confidences.values() or [0.5]))
    profit_est_12m = float(mgr.get("profit_est_12m") or 0.0)

    debate_scores = {
        "P_bull": round(float(P_bull), 4),
        "P_bear": round(float(P_bear), 4),
        "P_base": round(float(P_base), 4),
        "conviction": conviction,
        "profit_est_12m": profit_est_12m,
        "bull_agents": bull_agents,
        "bear_agents": bear_agents,
        "hold_agents": hold_agents,
        "uncertainty_score": 0.1,
    }

    # Token usage approximation (AG2 does not expose per-call easily)
    total_chars = sum(len((m.get("content") or "")) for m in messages or [])
    approx_tokens = total_chars // 4

    transcript = {
        "symbol": symbol,
        "company_name": company_name,
        "sector": sector,
        "langgraph_messages": langgraph_messages,
        "round1": [parsed_per_agent.get(a, {}) for a in ("growth", "value", "macro", "risk") if a in parsed_per_agent],
        "round2": [],
        "round3": [],
        "final_stances": (mgr.get("final_stance") or "HOLD").upper(),
        "token_usage": {"total_tokens": approx_tokens},
        "debate_scores": debate_scores,
        "conviction": conviction,
        "profit_est_12m": profit_est_12m,
        "engine": "ag2",
        "raw_chat_count": len(messages or []),
    }
    return transcript


# Public alias mirroring the LangGraph entrypoint
def run_debate_for_stock(symbol: str, company_name: str, sector: str, data_context: dict, max_workers: int = 4) -> dict:
    engine = AG2DebateEngine(max_workers=max_workers)
    return engine.run_debate(symbol, company_name, sector, data_context)
