"""
unit/test_transform_messages_advanced.py
测试 transform_messages 的高级边界情况：
- ThinkingContent 跨模型降级
- stop_reason='error' 的 AssistantMessage 被跳过
- 孤儿 ToolCall 自动补充 ToolResultMessage
- 多轮 Tool 对话上下文完整性
"""
import pytest
from provider.types import (
    Model, ModelCost, Context,
    UserMessage, AssistantMessage, ToolResultMessage,
    TextContent, ThinkingContent, ToolCall
)
from provider.api.transform_messages import transform_messages


def make_model(provider="openai", model_id="gpt-4o", api="openai-completions") -> Model:
    return Model(
        id=model_id, name="Test", api=api,
        provider=provider, base_url="http://test",
        reasoning=False, input=["text"],
        cost=ModelCost(0, 0, 0, 0),
        context_window=4096, max_tokens=1000
    )


class TestThinkingContentHandling:
    def test_thinking_block_preserved_for_same_model(self):
        """同模型：ThinkingContent 原样保留（即使有 thinking_signature）。"""
        m = make_model("openai", "gpt-4o")
        thinking = ThinkingContent(thinking="My thoughts", thinking_signature="sig-123")
        # 标记 AssistantMessage 为同一 provider/api/model
        asst = AssistantMessage(
            provider="openai", api="openai-completions", model="gpt-4o",
            content=[thinking, TextContent(text="Answer")]
        )
        result = transform_messages([UserMessage(content="Hi"), asst], m)
        asst_result = next(r for r in result if r.role == "assistant")
        thinking_blocks = [b for b in asst_result.content if isinstance(b, ThinkingContent)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking_signature == "sig-123"

    def test_thinking_block_downgraded_to_text_for_different_model(self):
        """跨模型：带内容的 ThinkingContent 被降级为 TextContent。"""
        # 消息来自 anthropic，但当前请求是 openai 模型
        asst = AssistantMessage(
            provider="anthropic", api="anthropic-messages", model="claude-3",
            content=[ThinkingContent(thinking="Deep thoughts"), TextContent(text="Answer")]
        )
        m = make_model("openai", "gpt-4o")
        result = transform_messages([UserMessage(content="Hi"), asst], m)
        asst_result = next(r for r in result if r.role == "assistant")
        thinking_blocks = [b for b in asst_result.content if isinstance(b, ThinkingContent)]
        text_blocks = [b for b in asst_result.content if isinstance(b, TextContent)]
        # 思考块应该被转成文本
        assert len(thinking_blocks) == 0
        assert any("Deep thoughts" in b.text for b in text_blocks)

    def test_empty_thinking_block_dropped(self):
        """空的 ThinkingContent（无内容）应被丢弃。"""
        asst = AssistantMessage(
            provider="anthropic", api="anthropic-messages", model="claude-3",
            content=[ThinkingContent(thinking=""), TextContent(text="Answer")]
        )
        m = make_model("openai", "gpt-4o")
        result = transform_messages([UserMessage(content="Hi"), asst], m)
        asst_result = next(r for r in result if r.role == "assistant")
        thinking_blocks = [b for b in asst_result.content if isinstance(b, ThinkingContent)]
        assert len(thinking_blocks) == 0

    def test_redacted_thinking_preserved_for_same_model(self):
        """同模型：redacted=True 的 ThinkingContent 保留（API 需要它）。"""
        m = make_model("anthropic", "claude-3", "anthropic-messages")
        thinking = ThinkingContent(thinking="", redacted=True, thinking_signature="enc-sig")
        asst = AssistantMessage(
            provider="anthropic", api="anthropic-messages", model="claude-3",
            content=[thinking, TextContent(text="Answer")]
        )
        result = transform_messages([UserMessage(content="Hi"), asst], m)
        asst_result = next(r for r in result if r.role == "assistant")
        thinking_blocks = [b for b in asst_result.content if isinstance(b, ThinkingContent)]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].redacted is True


class TestErroredAssistantMessageSkipped:
    def test_errored_assistant_message_is_skipped(self):
        """stop_reason='error' 的 AssistantMessage 在 second pass 中被丢弃。"""
        errored_asst = AssistantMessage(
            stop_reason="error",
            content=[TextContent(text="Something went wrong")]
        )
        msgs = [UserMessage(content="Hi"), errored_asst, UserMessage(content="Try again")]
        m = make_model()
        result = transform_messages(msgs, m)
        roles = [r.role for r in result]
        # errored_asst 被过滤掉
        assert roles.count("assistant") == 0
        assert roles.count("user") == 2

    def test_aborted_assistant_message_is_skipped(self):
        """stop_reason='aborted' 的 AssistantMessage 同样被丢弃。"""
        aborted_asst = AssistantMessage(
            stop_reason="aborted",
            content=[TextContent(text="Incomplete response")]
        )
        msgs = [UserMessage(content="Hi"), aborted_asst]
        m = make_model()
        result = transform_messages(msgs, m)
        roles = [r.role for r in result]
        assert "assistant" not in roles


class TestSyntheticToolResults:
    def test_orphan_tool_call_gets_synthetic_result(self):
        """ToolCall 没有对应的 ToolResultMessage 时，系统自动补充一条错误结果。"""
        tc = ToolCall(id="call_orphan", name="get_data", arguments={})
        asst = AssistantMessage(content=[tc])
        msgs = [UserMessage(content="Hi"), asst]
        m = make_model()
        result = transform_messages(msgs, m)

        assert len(result) == 3  # user + assistant + synthetic tool result
        synthetic = result[2]
        assert synthetic.role == "toolResult"
        assert synthetic.tool_call_id == "call_orphan"
        assert synthetic.is_error is True
        assert "No result provided" in synthetic.content[0].text

    def test_matched_tool_result_no_synthetic(self):
        """有对应 ToolResultMessage 时，不补充合成结果。"""
        tc = ToolCall(id="call_matched", name="get_data", arguments={})
        asst = AssistantMessage(content=[tc])
        tool_result = ToolResultMessage(
            tool_call_id="call_matched", tool_name="get_data",
            content=[TextContent(text="Some result")]
        )
        msgs = [UserMessage(content="Hi"), asst, tool_result]
        m = make_model()
        result = transform_messages(msgs, m)

        tool_results = [r for r in result if r.role == "toolResult"]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False

    def test_multiple_tool_calls_all_matched(self):
        """多个 ToolCall 全部有对应结果：不产生任何合成条目。"""
        tc1 = ToolCall(id="call_1", name="tool_a", arguments={})
        tc2 = ToolCall(id="call_2", name="tool_b", arguments={})
        asst = AssistantMessage(content=[tc1, tc2])
        tr1 = ToolResultMessage(tool_call_id="call_1", tool_name="tool_a", content=[TextContent(text="r1")])
        tr2 = ToolResultMessage(tool_call_id="call_2", tool_name="tool_b", content=[TextContent(text="r2")])
        msgs = [UserMessage(content="Hi"), asst, tr1, tr2]
        m = make_model()
        result = transform_messages(msgs, m)
        tool_results = [r for r in result if r.role == "toolResult"]
        # 只有 2 个真实结果，没有合成的
        assert len(tool_results) == 2
        assert all(not r.is_error for r in tool_results)

    def test_partial_tool_results_fills_missing(self):
        """2 个 ToolCall 只有 1 个结果时，补充 1 个合成错误结果。"""
        tc1 = ToolCall(id="call_1", name="tool_a", arguments={})
        tc2 = ToolCall(id="call_2", name="tool_b", arguments={})
        asst = AssistantMessage(content=[tc1, tc2])
        tr1 = ToolResultMessage(tool_call_id="call_1", tool_name="tool_a", content=[TextContent(text="r1")])
        msgs = [UserMessage(content="Hi"), asst, tr1]
        m = make_model()
        result = transform_messages(msgs, m)
        tool_results = [r for r in result if r.role == "toolResult"]
        assert len(tool_results) == 2
        error_results = [r for r in tool_results if r.is_error]
        assert len(error_results) == 1
        assert error_results[0].tool_call_id == "call_2"


class TestMultiRoundConversation:
    def test_multi_round_tool_use_preserved(self):
        """完整的多轮 Tool Use 对话结构被正确保留。"""
        # Round 1: user -> assistant (with tool call) -> tool result
        tc = ToolCall(id="call_r1", name="search", arguments={"q": "python"})
        asst1 = AssistantMessage(content=[TextContent(text="Let me search."), tc])
        tr1 = ToolResultMessage(tool_call_id="call_r1", tool_name="search",
                                content=[TextContent(text="Python results...")])
        # Round 2: assistant final answer
        asst2 = AssistantMessage(content=[TextContent(text="Here is the answer.")])

        msgs = [
            UserMessage(content="Search for python"),
            asst1,
            tr1,
            asst2,
            UserMessage(content="Thanks")
        ]
        m = make_model()
        result = transform_messages(msgs, m)

        roles = [r.role for r in result]
        assert roles == ["user", "assistant", "toolResult", "assistant", "user"]
