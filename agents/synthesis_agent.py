import json


import config
from agents.base_agent import BaseAgent
from llm.json_utils import extract_json


SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis engine and debate adjudicator.

Input: multiple bull-case and bear-case analyses for ONE stock.
Output: a compact, loss-minimized decision packet for scenario modeling.

Rules:
- Output MUST be valid JSON (no markdown fences).
- Maximize information density. No prose paragraphs.
- Prefer short bullets and numeric facts.
- De-duplicate aggressively across agents.
- If evidence is weak/unclear, say so explicitly in uncertainties.
- Explicitly resolve the debate: capture why bulls win, why bears win, and what would change the call.

Return JSON exactly in this shape:
{
  "symbol": "SYMBOL",
  "asof": "YYYY-MM-DD",
  "bull_points": ["..."],
  "bear_points": ["..."],
  "key_uncertainties": ["..."],
  "debate_log": {
    "bull_summary": ["..."],
    "bear_summary": ["..."],
    "resolution": "..."
  },
  "numeric_facts": {
    "price": 0,
    "pe": null,
    "roe": null,
    "revenue_growth": null,
    "earnings_growth": null
  },
  "catalysts_90d": ["..."],
  "risks": ["..."],
  "thesis_invalidation": "..."
}
"""

SYNTHESIS_VALIDATOR_PROMPT = """You are a loss-check validator for a compressed research packet.

You will receive:
- Raw bull/bear case texts (truncated) for one stock
- A synthesized packet JSON

Task:
- Identify any important unique claims present in raw texts that are missing from the packet.
- Only include HIGH-SIGNAL items: numeric facts, near-term catalysts, thesis-breaking risks, valuation/rating changes, regulatory actions, earnings guidance/events, major contracts.
- Do NOT repeat items already present (near-duplicates count as present).

Return JSON only:
{
  "missing_bull_points": ["..."],
  "missing_bear_points": ["..."],
  "missing_catalysts_90d": ["..."],
  "missing_risks": ["..."],
  "missing_uncertainties": ["..."],
  "missing_thesis_invalidation": "..."  // empty string if none
}
"""


class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            tools=[],
            model=config.MODEL_FAST,
            max_tokens=config.SYNTHESIS_AGENT_MAX_TOKENS,
        )

    def synthesize(
        self,
        *,
        symbol: str,
        company_name: str,
        sector: str,
        fundamentals_summary: dict,
        bull_cases: list[dict],
        bear_cases: list[dict],
    ) -> dict:
        # Keep the prompt small: we include the raw cases but cap the per-case chars.
        def _cap(text: str, n: int = 1600) -> str:
            t = (text or "").strip()
            return t[:n]

        payload = {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "fundamentals": fundamentals_summary,
            "bull_cases": [
                {"agent_id": c.get("agent_id"), "success": c.get("success"), "text": _cap(c.get("thesis"))}
                for c in bull_cases
            ],
            "bear_cases": [
                {"agent_id": c.get("agent_id"), "success": c.get("success"), "text": _cap(c.get("thesis"))}
                for c in bear_cases
            ],
        }

        raw = self.run("Synthesize into decision packet JSON.", context=payload)
        packet = self._parse_packet(symbol=symbol, raw=raw)
        packet = self._normalize_and_cap(symbol=symbol, packet=packet)

        # Second pass: loss check and inject missing high-signal items (capped).
        try:
            validator = BaseAgent(
                system_prompt=SYNTHESIS_VALIDATOR_PROMPT,
                tools=[],
                model=config.MODEL_FAST,
                max_tokens=config.SYNTHESIS_VALIDATOR_MAX_TOKENS,
            )
            vctx = {
                "symbol": symbol,
                "raw_bull_cases": payload["bull_cases"],
                "raw_bear_cases": payload["bear_cases"],
                "packet": packet,
            }
            vraw = validator.run("Find missing high-signal items and return JSON.", context=vctx)
            missing = self._parse_missing(vraw)
            packet = self._inject_missing(packet, missing)
            packet = self._normalize_and_cap(symbol=symbol, packet=packet)
            packet["_validator_token_usage"] = validator.get_token_usage()
        except Exception:
            pass

        return packet

    def _parse_packet(self, *, symbol: str, raw: str) -> dict:
        try:
            return extract_json(raw, expected=dict)
        except Exception:
            return {
                "symbol": symbol,
                "asof": "",
                "bull_points": [],
                "bear_points": [],
                "key_uncertainties": ["Synthesis parse failed"],
                "numeric_facts": {},
                "catalysts_90d": [],
                "risks": [],
                "thesis_invalidation": "",
                "raw": (raw or "")[:500],
            }

    def _parse_missing(self, raw: str) -> dict:
        try:
            obj = extract_json(raw, expected=dict)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _normalize_and_cap(self, *, symbol: str, packet: dict) -> dict:
        out = dict(packet or {})
        out["symbol"] = out.get("symbol") or symbol
        out.setdefault("bull_points", [])
        out.setdefault("bear_points", [])
        out.setdefault("key_uncertainties", [])
        out.setdefault("debate_log", {})
        out.setdefault("numeric_facts", {})
        out.setdefault("catalysts_90d", [])
        out.setdefault("risks", [])
        out.setdefault("thesis_invalidation", "")

        out["bull_points"] = _dedupe_list(out.get("bull_points", []))[: config.SYNTHESIS_MAX_BULL_POINTS]
        out["bear_points"] = _dedupe_list(out.get("bear_points", []))[: config.SYNTHESIS_MAX_BEAR_POINTS]
        out["catalysts_90d"] = _dedupe_list(out.get("catalysts_90d", []))[: config.SYNTHESIS_MAX_CATALYSTS_90D]
        out["risks"] = _dedupe_list(out.get("risks", []))[: config.SYNTHESIS_MAX_RISKS]
        out["key_uncertainties"] = _dedupe_list(out.get("key_uncertainties", []))[: config.SYNTHESIS_MAX_UNCERTAINTIES]
        out["thesis_invalidation"] = (out.get("thesis_invalidation") or "").strip()
        if not isinstance(out.get("debate_log"), dict):
            out["debate_log"] = {}
        if not isinstance(out.get("numeric_facts"), dict):
            out["numeric_facts"] = {}
        return out

    def _inject_missing(self, packet: dict, missing: dict) -> dict:
        if not missing:
            return packet
        out = dict(packet)

        inject_cap = config.SYNTHESIS_MAX_MISSING_INJECT
        out["bull_points"] = _inject(out.get("bull_points", []), missing.get("missing_bull_points", []), inject_cap)
        out["bear_points"] = _inject(out.get("bear_points", []), missing.get("missing_bear_points", []), inject_cap)
        out["catalysts_90d"] = _inject(out.get("catalysts_90d", []), missing.get("missing_catalysts_90d", []), inject_cap)
        out["risks"] = _inject(out.get("risks", []), missing.get("missing_risks", []), inject_cap)
        out["key_uncertainties"] = _inject(out.get("key_uncertainties", []), missing.get("missing_uncertainties", []), inject_cap)

        inv = (missing.get("missing_thesis_invalidation") or "").strip()
        if inv and not (out.get("thesis_invalidation") or "").strip():
            out["thesis_invalidation"] = inv
        return out


def _dedupe_list(items: list) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items or []:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        if s.lower() == "none":
            continue
        key = " ".join(s.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _inject(existing: list, additions: list, cap: int) -> list[str]:
    base = _dedupe_list(existing or [])
    add = _dedupe_list(additions or [])
    injected = 0
    for item in add:
        if injected >= cap:
            break
        key = " ".join(item.lower().split())
        if any(" ".join(e.lower().split()) == key for e in base):
            continue
        base.append(item)
        injected += 1
    return base
