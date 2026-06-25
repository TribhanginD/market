from __future__ import annotations

import json
import re
from typing import Any


def extract_json(value: str, *, expected: type | tuple[type, ...] = dict) -> Any:
    """
    Parse JSON from model output that may include thinking/prose or fences.
    Returns the first valid object/array matching `expected`, preferring later
    candidates because many reasoning models put analysis before the answer.
    """
    text = (value or "").strip()
    if not text:
        raise ValueError("empty JSON response")

    candidates = _candidate_json_strings(text)
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception as e:
            errors.append(str(e))
            continue
        if isinstance(parsed, expected):
            return parsed

    expected_name = (
        "/".join(t.__name__ for t in expected)
        if isinstance(expected, tuple)
        else expected.__name__
    )
    detail = errors[-1] if errors else "no JSON object or array found"
    raise ValueError(f"could not extract {expected_name} JSON: {detail}")


def _candidate_json_strings(text: str) -> list[str]:
    out: list[str] = []
    stripped = _strip_code_fence(text)
    out.append(stripped)

    fences = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    ]
    out.extend(reversed(fences))

    out.extend(reversed(_balanced_spans(text, "{", "}")))
    out.extend(reversed(_balanced_spans(text, "[", "]")))

    seen = set()
    deduped = []
    for item in out:
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _balanced_spans(text: str, opener: str, closer: str) -> list[str]:
    spans: list[str] = []
    stack = 0
    start: int | None = None
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opener:
            if stack == 0:
                start = idx
            stack += 1
        elif ch == closer and stack:
            stack -= 1
            if stack == 0 and start is not None:
                spans.append(text[start : idx + 1])
                start = None
    return spans
