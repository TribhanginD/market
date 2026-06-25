from __future__ import annotations

import json
import logging
import random
import time
import threading
from dataclasses import dataclass
from typing import Any, Optional

import requests

import config

logger = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Optional[dict[str, Any]] = None
    # Set when model wants to call tools (Gemini function-calling path)
    tool_calls: Optional[list[dict[str, Any]]] = None  # [{"name": ..., "args": {...}, "id": ...}]
    finish_reason: str = "stop"  # "stop" | "tool_calls"


_PROVIDER_PREFIX_ALIASES = {
    "groq": "groq",
    "gemini": "gemini",
    "mistral": "mistral",
    "openai": "openai",
    "openai-compat": "openai",
    "openai_compat": "openai",
    "openai-compatible": "openai",
    "compat": "openai",
}


def _looks_like_anthropic_model(model: str) -> bool:
    name = (model or "").strip().lower()
    return name.startswith(("claude-", "anthropic."))


def parse_model_provider(model: str) -> tuple[str, str]:
    """
    Allows selecting providers by prefix:
    - `groq:<model>`
    - `gemini:<model>`
    - `mistral:<model>`
    - `openai:<model>` (OpenAI-compatible `/v1/chat/completions` base URL)
    - default: anthropic

    If OPENAI_COMPAT_BASE_URL is set, unprefixed non-Claude model names are also
    routed to the OpenAI-compatible client. This lets `.env` contain the raw
    model id exposed by a local/self-hosted endpoint.
    """
    m = (model or "").strip()
    if ":" in m:
        prefix, rest = m.split(":", 1)
        prefix = prefix.strip().lower()
        rest = rest.strip()
        provider = _PROVIDER_PREFIX_ALIASES.get(prefix)
        if provider:
            return provider, rest
    if m and config.OPENAI_COMPAT_BASE_URL and not _looks_like_anthropic_model(m):
        return "openai", m
    return "anthropic", m


def normalize_openai_compat_base_url(base_url: str | None) -> str:
    value = (base_url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/models"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].rstrip("/")
            break
    return value


class OpenAICompatProvider:
    """
    Minimal OpenAI-compatible chat-completions client.
    Useful for self-hosted endpoints that expose `/v1/chat/completions`.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout_s: int | None = None):
        self.api_key = api_key if api_key is not None else config.OPENAI_COMPAT_API_KEY
        self.base_url = normalize_openai_compat_base_url(
            base_url if base_url is not None else config.OPENAI_COMPAT_BASE_URL
        )
        self.timeout_s = timeout_s if timeout_s is not None else config.LLM_HTTP_TIMEOUT_SECONDS

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _extra_payload(self, model: str) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        configured = getattr(config, "OPENAI_COMPAT_EXTRA_BODY_JSON", "")
        if configured:
            try:
                parsed = json.loads(configured)
                if isinstance(parsed, dict):
                    extra.update(parsed)
            except Exception as e:
                raise ValueError("OPENAI_COMPAT_EXTRA_BODY_JSON must be valid JSON object") from e

        model_l = (model or "").lower()
        if getattr(config, "OPENAI_COMPAT_DISABLE_QWEN_THINKING", True) and "qwen" in model_l:
            chat_template_kwargs = dict(extra.get("chat_template_kwargs") or {})
            chat_template_kwargs.setdefault("enable_thinking", False)
            extra["chat_template_kwargs"] = chat_template_kwargs
        return extra

    def chat(self, *, model: str, system: str, messages: list[dict], max_tokens: int | None, temperature: float) -> ProviderResponse:
        if not self.base_url:
            raise ValueError("OPENAI_COMPAT_BASE_URL is not set (expected something like https://host/v1).")

        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": payload_messages,
            "temperature": temperature,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        payload.update(self._extra_payload(model))

        data = self._post_with_retries(
            url=self._chat_completions_url(),
            headers=headers,
            payload=payload,
        )

        text = self._extract_text(data)
        usage = data.get("usage") or {}
        return ProviderResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            raw=data if isinstance(data, dict) else None,
        )

    def _extract_text(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"].get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(str(item.get("text", "")))
                        elif "text" in item:
                            parts.append(str(item.get("text", "")))
                return "".join(parts)
        except Exception:
            pass
        return ""

    def _post_with_retries(self, *, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        max_retries = getattr(config, "LLM_MAX_RETRIES", 6)
        base = getattr(config, "LLM_RETRY_BASE_SECONDS", 1.0)
        cap = getattr(config, "LLM_RETRY_MAX_SECONDS", 20.0)
        transient_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                if resp.status_code in transient_statuses and attempt < max_retries:
                    sleep_s = min(cap, base * (2 ** attempt)) * (0.8 + random.random() * 0.4)
                    logger.warning("OpenAI-compat transient error status=%s; retrying in %.1fs", resp.status_code, sleep_s)
                    time.sleep(sleep_s)
                    continue

                try:
                    resp.raise_for_status()
                except requests.HTTPError as e:
                    body = ""
                    try:
                        body = (resp.text or "")[:800]
                    except Exception:
                        body = ""
                    raise RuntimeError(f"OpenAI-compat HTTP {resp.status_code}: {body}") from e

                payload_json = resp.json()
                return payload_json if isinstance(payload_json, dict) else {"raw": payload_json}
            except RuntimeError:
                raise
            except requests.RequestException as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                sleep_s = min(cap, base * (2 ** attempt)) * (0.8 + random.random() * 0.4)
                logger.warning("OpenAI-compat network error; retrying in %.1fs (%s)", sleep_s, str(e)[:120])
                time.sleep(sleep_s)
                continue

        raise RuntimeError(f"OpenAI-compat request failed after retries. url={url}") from last_exc


class GroqProvider:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout_s: int | None = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.base_url = (base_url or config.GROQ_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else config.LLM_HTTP_TIMEOUT_SECONDS

    def chat(self, *, model: str, system: str, messages: list[dict], max_tokens: int | None, temperature: float) -> ProviderResponse:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        # Groq is OpenAI-compatible; include system as a system message.
        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        payload = {
            "model": model,
            "messages": payload_messages,
            "temperature": temperature,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()

        text = ""
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except Exception:
            text = ""

        usage = data.get("usage") or {}
        return ProviderResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            raw=data if isinstance(data, dict) else None,
        )


class MistralProvider:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout_s: int | None = None):
        self.api_key = api_key or config.MISTRAL_API_KEY
        self.base_url = (base_url or config.MISTRAL_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else config.LLM_HTTP_TIMEOUT_SECONDS

    def chat(self, *, model: str, system: str, messages: list[dict], max_tokens: int | None, temperature: float) -> ProviderResponse:
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set.")

        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        payload = {
            "model": model,
            "messages": payload_messages,
            "temperature": temperature,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        data = self._post_with_retries(
            url=f"{self.base_url}/chat/completions",
            payload=payload,
        )

        text = ""
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif "text" in item:
                            parts.append(str(item.get("text", "")))
                text = "".join(parts)
        except Exception:
            text = ""

        usage = data.get("usage") or {}
        return ProviderResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            raw=data if isinstance(data, dict) else None,
        )

    def _post_with_retries(self, *, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        max_retries = getattr(config, "LLM_MAX_RETRIES", 6)
        base = getattr(config, "LLM_RETRY_BASE_SECONDS", 1.0)
        cap = getattr(config, "LLM_RETRY_MAX_SECONDS", 20.0)
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt >= max_retries:
                        break
                    sleep_s = min(cap, base * (2 ** attempt)) * (0.8 + random.random() * 0.4)
                    logger.warning("Mistral transient error status=%s; retrying in %.1fs", resp.status_code, sleep_s)
                    time.sleep(sleep_s)
                    continue
                resp.raise_for_status()
                payload_json = resp.json()
                return payload_json if isinstance(payload_json, dict) else {"raw": payload_json}
            except requests.HTTPError as e:
                last_exc = e
                body = ""
                try:
                    body = (resp.text or "")[:800]
                except Exception:
                    body = ""
                raise RuntimeError(f"Mistral HTTP {resp.status_code}: {body}") from e
            except requests.RequestException as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                sleep_s = min(cap, base * (2 ** attempt)) * (0.8 + random.random() * 0.4)
                logger.warning("Mistral network error; retrying in %.1fs (%s)", sleep_s, str(e)[:120])
                time.sleep(sleep_s)
                continue

        raise RuntimeError(f"Mistral request failed after retries. url={url}") from last_exc


class _GeminiRateLimiter:
    """Token-bucket rate limiter: max 15 requests per 60s window."""
    def __init__(self, rpm: int = 15):
        self._rpm = rpm
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps older than 60s
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return
                wait = 60.0 - (now - self._timestamps[0]) + 0.05
            logger.debug("Gemini RPM limit hit; sleeping %.1fs", wait)
            time.sleep(max(wait, 0.1))


_gemini_limiter = _GeminiRateLimiter(rpm=15)


class GeminiProvider:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout_s: int | None = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.base_url = (base_url or config.GEMINI_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else config.LLM_HTTP_TIMEOUT_SECONDS

    @staticmethod
    def _anthropic_tools_to_gemini(tools: list[dict]) -> list[dict] | None:
        """Convert Anthropic tool format → Gemini function_declarations."""
        if not tools:
            return None
        decls = []
        for t in tools:
            schema = t.get("input_schema") or {}
            decl: dict[str, Any] = {"name": t["name"], "description": t.get("description", "")}
            params: dict[str, Any] = {"type": "object"}
            props = schema.get("properties")
            if props:
                params["properties"] = props
            req = schema.get("required")
            if req:
                params["required"] = req
            decl["parameters"] = params
            decls.append(decl)
        return [{"function_declarations": decls}]

    def chat(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float,
        tools: list[dict] | None = None,
        gemini_contents: list[dict] | None = None,  # pass raw Gemini contents for multi-turn
    ) -> ProviderResponse:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        if gemini_contents is not None:
            contents = gemini_contents
        else:
            contents = []
            for m in messages:
                role = m.get("role")
                if role == "assistant":
                    role = "model"
                if role not in ("user", "model"):
                    continue
                content = m.get("content", "")
                if isinstance(content, str):
                    contents.append({"role": role, "parts": [{"text": content}]})
                elif isinstance(content, list):
                    # Anthropic-style content blocks → text only
                    text = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    if text:
                        contents.append({"role": role, "parts": [{"text": text}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if max_tokens is not None and max_tokens > 0:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        gemini_tools = self._anthropic_tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        _gemini_limiter.acquire()
        data = self._post_with_retries(url=url, payload=payload)

        candidates = (data.get("candidates") or []) if isinstance(data, dict) else []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        finish_reason_raw = (candidates[0].get("finishReason") or "STOP") if candidates else "STOP"

        # Detect function calls
        tool_calls: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            if "functionCall" in p:
                fc = p["functionCall"]
                tool_calls.append({
                    "name": fc.get("name", ""),
                    "args": fc.get("args") or {},
                    "id": fc.get("name", ""),  # Gemini has no call ID; use name as key
                })
            elif "text" in p:
                text_parts.append(p["text"])

        text = "".join(text_parts)
        finish_reason = "tool_calls" if tool_calls else "stop"

        usage = data.get("usageMetadata") if isinstance(data, dict) else None
        input_tokens = 0
        output_tokens = 0
        try:
            if isinstance(usage, dict):
                input_tokens = int(usage.get("promptTokenCount") or 0)
                output_tokens = int(usage.get("candidatesTokenCount") or 0)
        except Exception:
            pass

        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            raw=data if isinstance(data, dict) else None,
        )

    def _post_with_retries(self, *, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        max_retries = getattr(config, "LLM_MAX_RETRIES", 6)
        base = getattr(config, "LLM_RETRY_BASE_SECONDS", 1.0)
        cap = getattr(config, "LLM_RETRY_MAX_SECONDS", 20.0)

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    # Retryable server-side/transient errors.
                    if attempt >= max_retries:
                        break
                    sleep_s = min(cap, base * (2 ** attempt)) * (0.8 + random.random() * 0.4)
                    logger.warning("Gemini transient error status=%s; retrying in %.1fs", resp.status_code, sleep_s)
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                payload_json = resp.json()
                return payload_json if isinstance(payload_json, dict) else {"raw": payload_json}
            except requests.HTTPError as e:
                last_exc = e
                # Non-retryable HTTP error.
                break
            except requests.RequestException as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                sleep_s = min(cap, base * (2 ** attempt)) * (0.8 + random.random() * 0.4)
                logger.warning("Gemini network error; retrying in %.1fs (%s)", sleep_s, str(e)[:120])
                time.sleep(sleep_s)
                continue

        # Sanitize key from URL in any raised exception.
        safe_url = url.split("?", 1)[0]
        raise RuntimeError(f"Gemini request failed after retries. url={safe_url}") from last_exc
