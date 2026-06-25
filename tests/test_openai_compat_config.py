import importlib

import config
from llm.providers import (
    OpenAICompatProvider,
    normalize_openai_compat_base_url,
    parse_model_provider,
)


def test_stage_model_env_overrides_are_honored(monkeypatch):
    try:
        with monkeypatch.context() as m:
            m.setenv("MODEL_FAST", "openai:fast-model")
            m.setenv("MODEL_SMART", "openai:smart-model")
            m.setenv("MODEL_BEST", "openai:best-model")
            m.setenv("STAGE1_MODEL", "openai:stage1-model")
            m.setenv("STAGE2_MODEL", "openai:stage2-model")
            m.setenv("STAGE3_MODEL", "openai:stage3-model")
            m.setenv("STAGE4_MODEL", "openai:stage4-model")
            m.setenv("STAGE5_MODEL", "openai:stage5-model")

            cfg = importlib.reload(config)

            assert cfg.MODEL_FAST == "openai:fast-model"
            assert cfg.MODEL_SMART == "openai:smart-model"
            assert cfg.MODEL_BEST == "openai:best-model"
            assert cfg.STAGE1_MODEL == "openai:stage1-model"
            assert cfg.STAGE2_MODEL == "openai:stage2-model"
            assert cfg.STAGE3_MODEL == "openai:stage3-model"
            assert cfg.STAGE4_MODEL == "openai:stage4-model"
            assert cfg.STAGE5_MODEL == "openai:stage5-model"
    finally:
        importlib.reload(config)


def test_openai_compat_model_replaces_default_claude_models(monkeypatch):
    """When OPENAI_COMPAT_MODEL is set and no explicit per-model env vars are
    provided, defaults route through the OpenAI-compat endpoint. Explicit env
    values are always respected (see test_stage_model_env_overrides_are_honored)."""
    try:
        with monkeypatch.context() as m:
            m.setenv("OPENAI_COMPAT_MODEL", "served-model")
            # Setenv to empty (rather than delenv) because config.load_dotenv()
            # re-populates from `.env` on reload; setenv("") overrides .env.
            for var in (
                "MODEL_FAST", "MODEL_SMART", "MODEL_BEST",
                "STAGE1_MODEL", "STAGE2_MODEL", "STAGE3_MODEL",
                "STAGE4_MODEL", "STAGE5_MODEL",
            ):
                m.setenv(var, "")

            cfg = importlib.reload(config)

            assert cfg.MODEL_FAST == "openai:served-model"
            assert cfg.MODEL_SMART == "openai:served-model"
            assert cfg.STAGE1_MODEL == "openai:served-model"
            assert cfg.STAGE2_MODEL == "openai:served-model"
    finally:
        importlib.reload(config)


def test_explicit_model_env_not_overridden_by_compat(monkeypatch):
    """Explicit user-set STAGE*_MODEL must NOT be silently downgraded by
    OPENAI_COMPAT_MODEL — regression for P2.7 fix."""
    try:
        with monkeypatch.context() as m:
            m.setenv("OPENAI_COMPAT_MODEL", "served-model")
            m.setenv("STAGE4_MODEL", "claude-opus-4-5")

            cfg = importlib.reload(config)

            assert cfg.STAGE4_MODEL == "claude-opus-4-5"
    finally:
        importlib.reload(config)


def test_raw_model_routes_to_openai_compat_when_base_url_set(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_COMPAT_BASE_URL", "https://cloudspace.example/v1")

    assert parse_model_provider("gemma-2-9b-it") == ("openai", "gemma-2-9b-it")
    assert parse_model_provider("claude-3-haiku-20240307") == ("anthropic", "claude-3-haiku-20240307")


def test_openai_compat_base_url_accepts_full_chat_completions_url():
    assert (
        normalize_openai_compat_base_url(" https://cloudspace.example/v1/chat/completions/ ")
        == "https://cloudspace.example/v1"
    )
    assert normalize_openai_compat_base_url("https://cloudspace.example/v1/models") == "https://cloudspace.example/v1"

    provider = OpenAICompatProvider(base_url="https://cloudspace.example/v1/chat/completions")
    assert provider._chat_completions_url() == "https://cloudspace.example/v1/chat/completions"


def test_qwen_openai_compat_disables_thinking_by_default(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_COMPAT_EXTRA_BODY_JSON", "")
    monkeypatch.setattr(config, "OPENAI_COMPAT_DISABLE_QWEN_THINKING", True)

    provider = OpenAICompatProvider(base_url="https://cloudspace.example/v1")

    assert provider._extra_payload("Qwen/Qwen3.5-9B") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_openai_compat_omits_max_tokens_when_zero(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_COMPAT_EXTRA_BODY_JSON", "")
    monkeypatch.setattr(config, "OPENAI_COMPAT_DISABLE_QWEN_THINKING", False)

    provider = OpenAICompatProvider(base_url="https://cloudspace.example/v1")
    captured_payload = {}

    def fake_post(*, url, headers, payload):
        captured_payload.update(payload)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(provider, "_post_with_retries", fake_post)

    provider.chat(
        model="Qwen/Qwen3.5-9B",
        system="",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=0,
        temperature=0.1,
    )

    assert "max_tokens" not in captured_payload
