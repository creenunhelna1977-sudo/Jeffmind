"""
unit/test_normalize_tool_call_id.py
测试工具调用 ID 清洗逻辑。
"""
from provider.types import Model, ModelCost, AssistantMessage
from provider.api.openai_completions import _normalize_tool_call_id


def make_model(provider: str = "openai") -> Model:
    return Model(
        id="test", name="Test", api="openai-completions",
        provider=provider, base_url="https://api.openai.com",
        reasoning=False, input=["text"],
        cost=ModelCost(0, 0, 0, 0),
        context_window=4096, max_tokens=1000
    )


def test_plain_id_passthrough():
    """普通 ID 不含管道符：原样返回。"""
    m = make_model()
    result = _normalize_tool_call_id("call_abc123", m, AssistantMessage())
    assert result == "call_abc123"


def test_pipe_separated_id_takes_first_part():
    """含管道符的 ID：取第一段并清洗。"""
    m = make_model()
    result = _normalize_tool_call_id("call_abc|extra_info", m, AssistantMessage())
    assert result == "call_abc"


def test_pipe_id_sanitizes_special_chars():
    """管道前部分含特殊字符：全部替换为下划线。"""
    m = make_model()
    result = _normalize_tool_call_id("call!@#valid_id|other", m, AssistantMessage())
    # 只保留 alnum、- 和 _
    assert all(c.isalnum() or c in "-_" for c in result)


def test_openai_provider_truncates_long_id():
    """OpenAI 提供商：超过 40 字符的 ID 需截断。"""
    m = make_model("openai")
    long_id = "a" * 50
    result = _normalize_tool_call_id(long_id, m, AssistantMessage())
    assert len(result) <= 40


def test_non_openai_provider_keeps_long_id():
    """非 OpenAI 提供商：不截断长 ID。"""
    m = make_model("deepseek")
    long_id = "a" * 50
    result = _normalize_tool_call_id(long_id, m, AssistantMessage())
    assert result == long_id


def test_pipe_id_truncated_to_40():
    """管道符 ID 清洗后最多 40 字符。"""
    m = make_model("openai")
    long_pipe_id = "a" * 50 + "|suffix"
    result = _normalize_tool_call_id(long_pipe_id, m, AssistantMessage())
    assert len(result) <= 40
