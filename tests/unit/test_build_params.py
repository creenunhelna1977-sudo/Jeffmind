"""
unit/test_build_params.py
测试 _build_params 最终 payload 组装逻辑。
"""
import pytest
from provider.types import Model, ModelCost, Context, UserMessage, Tool, StreamOptions
from provider.api.openai_completions import _build_params


def make_model(provider="openai", reasoning=False, compat=None) -> Model:
    return Model(
        id="gpt-4o", name="GPT-4o", api="openai-completions",
        provider=provider, base_url="https://api.openai.com/v1",
        reasoning=reasoning, input=["text"],
        cost=ModelCost(0, 0, 0, 0),
        context_window=4096, max_tokens=4096,
        compat=compat or {}
    )


def openai_compat():
    return {
        "supports_usage_in_streaming": True,
        "supports_store": True,
        "supports_developer_role": True,
        "supports_reasoning_effort": True,
        "supports_strict_mode": True,
        "supports_long_cache_retention": True,
        "max_tokens_field": "max_completion_tokens",
        "requires_tool_result_name": False,
        "requires_assistant_after_tool_result": False,
        "requires_thinking_as_text": False,
        "requires_reasoning_content_on_assistant_messages": False,
        "thinking_format": "openai",
    }


def deepseek_compat():
    return {
        **openai_compat(),
        "supports_store": False,
        "supports_developer_role": False,
        "supports_usage_in_streaming": True,
        "max_tokens_field": "max_completion_tokens",
        "thinking_format": "deepseek",
        "requires_reasoning_content_on_assistant_messages": True,
    }


class TestBuildParamsBasic:
    def test_always_includes_model_and_stream(self):
        m = make_model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        params = _build_params(m, ctx, None, openai_compat())
        assert params["model"] == "gpt-4o"
        assert params["stream"] is True

    def test_stream_options_included_when_supported(self):
        m = make_model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        params = _build_params(m, ctx, None, openai_compat())
        assert "stream_options" in params
        assert params["stream_options"]["include_usage"] is True

    def test_stream_options_excluded_when_unsupported(self):
        compat = {**openai_compat(), "supports_usage_in_streaming": False}
        m = make_model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        params = _build_params(m, ctx, None, compat)
        assert "stream_options" not in params

    def test_store_false_when_supported(self):
        m = make_model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        params = _build_params(m, ctx, None, openai_compat())
        assert params["store"] is False

    def test_store_excluded_when_unsupported(self):
        compat = {**openai_compat(), "supports_store": False}
        m = make_model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        params = _build_params(m, ctx, None, compat)
        assert "store" not in params


class TestBuildParamsOptions:
    def test_max_tokens_uses_correct_field_openai(self):
        m = make_model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        opts = StreamOptions(max_tokens=512)
        params = _build_params(m, ctx, opts, openai_compat())
        assert "max_completion_tokens" in params
        assert params["max_completion_tokens"] == 512
        assert "max_tokens" not in params

    def test_max_tokens_uses_max_tokens_field_for_deepseek(self):
        compat = {**deepseek_compat(), "max_tokens_field": "max_tokens"}
        m = make_model("deepseek")
        ctx = Context(messages=[UserMessage(content="Hi")])
        opts = StreamOptions(max_tokens=256)
        params = _build_params(m, ctx, opts, compat)
        assert "max_tokens" in params
        assert params["max_tokens"] == 256

    def test_temperature_included_when_set(self):
        m = make_model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        opts = StreamOptions(temperature=0.7)
        params = _build_params(m, ctx, opts, openai_compat())
        assert params["temperature"] == 0.7

    def test_temperature_excluded_when_not_set(self):
        m = make_model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        params = _build_params(m, ctx, None, openai_compat())
        assert "temperature" not in params


class TestBuildParamsTools:
    def test_tools_included_when_context_has_tools(self):
        m = make_model()
        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {}, "required": []}
        )
        ctx = Context(messages=[UserMessage(content="Hi")], tools=[tool])
        params = _build_params(m, ctx, None, openai_compat())
        assert "tools" in params
        assert len(params["tools"]) == 1
        assert params["tools"][0]["function"]["name"] == "get_weather"

    def test_tools_excluded_when_no_tools(self):
        m = make_model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        params = _build_params(m, ctx, None, openai_compat())
        assert "tools" not in params

    def test_strict_mode_included_when_supported(self):
        m = make_model()
        tool = Tool(name="t", description="d", parameters={})
        ctx = Context(messages=[UserMessage(content="Hi")], tools=[tool])
        params = _build_params(m, ctx, None, openai_compat())
        assert params["tools"][0]["function"]["strict"] is False

    def test_strict_mode_excluded_when_unsupported(self):
        compat = {**openai_compat(), "supports_strict_mode": False}
        m = make_model()
        tool = Tool(name="t", description="d", parameters={})
        ctx = Context(messages=[UserMessage(content="Hi")], tools=[tool])
        params = _build_params(m, ctx, None, compat)
        assert "strict" not in params["tools"][0]["function"]


class TestBuildParamsSystemPrompt:
    def test_system_role_for_non_reasoning(self):
        m = make_model(reasoning=False)
        ctx = Context(messages=[UserMessage(content="Hi")], system_prompt="Be helpful")
        params = _build_params(m, ctx, None, openai_compat())
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][0]["content"] == "Be helpful"

    def test_developer_role_for_reasoning_model_openai(self):
        m = make_model(reasoning=True)
        ctx = Context(messages=[UserMessage(content="Hi")], system_prompt="Be helpful")
        compat = {**openai_compat(), "supports_developer_role": True}
        params = _build_params(m, ctx, None, compat)
        assert params["messages"][0]["role"] == "developer"

    def test_system_role_when_developer_not_supported(self):
        m = make_model(reasoning=True)
        ctx = Context(messages=[UserMessage(content="Hi")], system_prompt="Be helpful")
        compat = {**openai_compat(), "supports_developer_role": False}
        params = _build_params(m, ctx, None, compat)
        assert params["messages"][0]["role"] == "system"
