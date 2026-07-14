"""
unit/test_compat.py
测试 _detect_compat 提供商识别逻辑和 _get_compat 覆盖合并。
"""
import pytest
from provider.types import Model, ModelCost

from provider.api.openai_completions import _detect_compat, _get_compat


def make_model(provider: str, base_url: str = "https://api.example.com", reasoning: bool = False, compat: dict = None) -> Model:
    return Model(
        id="test", name="Test", api="openai-completions",
        provider=provider, base_url=base_url,
        reasoning=reasoning, input=["text"],
        cost=ModelCost(0, 0, 0, 0),
        context_window=4096, max_tokens=1000,
        compat=compat or {}
    )


class TestDetectCompat:
    def test_deepseek_via_url(self):
        m = make_model("custom", base_url="https://api.deepseek.com/v1")
        compat = _detect_compat(m)
        assert compat["thinking_format"] == "deepseek"
        # deepseek 是非标准提供商，不支持 store
        assert compat["supports_store"] is False
        assert compat["supports_developer_role"] is False

    def test_deepseek_via_provider(self):
        m = make_model("deepseek")
        compat = _detect_compat(m)
        assert compat["thinking_format"] == "deepseek"
        assert compat["requires_reasoning_content_on_assistant_messages"] is True

    def test_openai_standard(self):
        m = make_model("openai", base_url="https://api.openai.com/v1")
        compat = _detect_compat(m)
        assert compat["supports_store"] is True
        assert compat["supports_developer_role"] is True
        assert compat["thinking_format"] == "openai"
        # openai 支持 max_completion_tokens
        assert compat["max_tokens_field"] == "max_completion_tokens"

    def test_moonshot_max_tokens_field(self):
        m = make_model("moonshotai", base_url="https://api.moonshot.cn/v1")
        compat = _detect_compat(m)
        # moonshot 使用 max_tokens 字段（非标准）
        assert compat["max_tokens_field"] == "max_tokens"
        # moonshot 没有特定的 thinking_format 识别，使用默认 'openai'
        assert compat["thinking_format"] == "openai"

    def test_openrouter(self):
        m = make_model("openrouter", base_url="https://openrouter.ai/api/v1")
        compat = _detect_compat(m)
        assert compat["thinking_format"] == "openrouter"
        # openrouter 不支持 developer role
        assert compat["supports_developer_role"] is False

    def test_nvidia_via_url(self):
        m = make_model("custom", base_url="https://integrate.api.nvidia.com/v1")
        compat = _detect_compat(m)
        assert compat["supports_strict_mode"] is False
        assert compat["supports_store"] is False

    def test_together_ai(self):
        m = make_model("together", base_url="https://api.together.ai/v1")
        compat = _detect_compat(m)
        assert compat["max_tokens_field"] == "max_tokens"
        assert compat["thinking_format"] == "together"


class TestGetCompatOverride:
    def test_model_compat_overrides_detected(self):
        """model.compat 字段可以覆盖自动识别的任何值。"""
        m = make_model("deepseek", compat={"thinking_format": "custom_format", "supports_store": True})
        compat = _get_compat(m)
        # 覆盖生效
        assert compat["thinking_format"] == "custom_format"
        assert compat["supports_store"] is True
        # 其他字段仍为自动识别的值
        assert compat["requires_reasoning_content_on_assistant_messages"] is True

    def test_empty_compat_uses_detected(self):
        m = make_model("openai")
        compat = _get_compat(m)
        assert compat["supports_store"] is True
