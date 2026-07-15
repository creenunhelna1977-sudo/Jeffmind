"""
agent/react.py

ReAct (Reasoning + Acting) 循环引擎，基于 provider 层构建。

循环逻辑：
  1. 把当前 Context 发给模型
  2. 若模型决定调用工具 (stop_reason == "toolUse")：
       a. yield ReActToolCallEvent（通知调用者）
       b. 执行工具函数，得到字符串结果
       c. 把 AssistantMessage + ToolResultMessage 追加进 context
       d. 继续下一轮
  3. 若模型停止 (stop_reason == "stop" / "length")：
       yield ReActDoneEvent，退出循环
  4. 若出错 / 超出最大迭代次数：
       yield ReActErrorEvent，退出循环
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable

from provider.models import Models
from provider.types import (
    AssistantMessage,
    Context,
    Model,
    StreamEvent,
    StreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
)


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------

@dataclass
class ReActTool:
    """
    封装一个可被 ReAct 引擎调用的工具。

    fn: 异步函数，接收 **arguments（模型传过来的 JSON 参数），返回字符串结果。
    """
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Awaitable[str]]

    def to_provider_tool(self) -> Tool:
        """转换为 provider 层使用的 Tool 类型。"""
        return Tool(name=self.name, description=self.description, parameters=self.parameters)


# ---------------------------------------------------------------------------
# ReAct 专属事件
# ---------------------------------------------------------------------------

@dataclass
class ReActIterationEvent:
    """每一轮迭代开始时发出。"""
    type: str = "react_iteration"
    iteration: int = 0
    message_count: int = 0   # 当前 context 中的消息数量


@dataclass
class ReActToolCallEvent:
    """模型决定调用某个工具时发出（执行前）。"""
    type: str = "react_tool_call"
    iteration: int = 0
    tool_call: ToolCall = field(default_factory=ToolCall)


@dataclass
class ReActToolResultEvent:
    """工具执行完成时发出（包含结果）。"""
    type: str = "react_tool_result"
    iteration: int = 0
    tool_call: ToolCall = field(default_factory=ToolCall)
    result: str = ""
    is_error: bool = False


@dataclass
class ReActStreamEvent:
    """把底层 provider StreamEvent 透传上来（可选，用于显示流式思考过程）。"""
    type: str = "react_stream"
    event: StreamEvent = None


@dataclass
class ReActDoneEvent:
    """循环正常结束，包含最终的 AssistantMessage。"""
    type: str = "react_done"
    iterations: int = 0
    message: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ReActErrorEvent:
    """循环因错误或超限而终止。"""
    type: str = "react_error"
    iteration: int = 0
    reason: str = ""
    message: AssistantMessage | None = None


ReActEvent = (
    ReActIterationEvent
    | ReActToolCallEvent
    | ReActToolResultEvent
    | ReActStreamEvent
    | ReActDoneEvent
    | ReActErrorEvent
)


# ---------------------------------------------------------------------------
# ReAct 引擎
# ---------------------------------------------------------------------------

class ReActAgent:
    """
    ReAct 引擎。

    Usage:
        agent = ReActAgent(models, model, tools, max_iterations=8)
        async for event in agent.run(context):
            ...
    """

    def __init__(
        self,
        models: Models,
        model: Model,
        tools: list[ReActTool],
        max_iterations: int = 10,
        options: StreamOptions | None = None,
        pass_through_stream_events: bool = False,
    ):
        self._models = models
        self._model = model
        self._tools = {t.name: t for t in tools}
        self._max_iterations = max_iterations
        self._options = options
        self._pass_through = pass_through_stream_events

    async def run(self, context: Context) -> AsyncGenerator[ReActEvent, None]:
        """
        运行 ReAct 循环，异步生成事件流。

        context 会被就地修改（追加消息历史），调用方可在 ReActDoneEvent 之后检查它。
        """
        # 确保 context 知道有哪些工具
        context = Context(
            messages=list(context.messages),
            system_prompt=context.system_prompt,
            tools=[t.to_provider_tool() for t in self._tools.values()],
        )

        for iteration in range(1, self._max_iterations + 1):
            yield ReActIterationEvent(
                iteration=iteration,
                message_count=len(context.messages)
            )

            # ── 1. 调用模型 ──────────────────────────────────────────────
            assistant_message: AssistantMessage | None = None

            async for event in self._models.stream(self._model, context, self._options):
                if self._pass_through:
                    yield ReActStreamEvent(event=event)

                if event.type == "done":
                    assistant_message = event.message
                elif event.type == "error":
                    yield ReActErrorEvent(
                        iteration=iteration,
                        reason=event.error.error_message or "unknown error",
                        message=event.error,
                    )
                    return

            if assistant_message is None:
                yield ReActErrorEvent(
                    iteration=iteration,
                    reason="Stream ended without a done event",
                )
                return

            # ── 2. 检查停止原因 ──────────────────────────────────────────
            if assistant_message.stop_reason in ("stop", "length"):
                yield ReActDoneEvent(
                    iterations=iteration,
                    message=assistant_message,
                )
                return

            if assistant_message.stop_reason != "toolUse":
                yield ReActErrorEvent(
                    iteration=iteration,
                    reason=f"Unexpected stop_reason: {assistant_message.stop_reason}",
                    message=assistant_message,
                )
                return

            # ── 3. 执行工具调用 ──────────────────────────────────────────
            tool_calls = [b for b in assistant_message.content if isinstance(b, ToolCall)]
            if not tool_calls:
                yield ReActErrorEvent(
                    iteration=iteration,
                    reason="stop_reason is toolUse but no ToolCall blocks found",
                    message=assistant_message,
                )
                return

            # 把 assistant 消息追加进历史
            context.messages.append(assistant_message)

            for tc in tool_calls:
                yield ReActToolCallEvent(iteration=iteration, tool_call=tc)

                tool = self._tools.get(tc.name)
                if tool is None:
                    result_text = f"Error: tool '{tc.name}' is not registered."
                    is_error = True
                else:
                    try:
                        result_text = await tool.fn(**tc.arguments)
                        is_error = False
                    except Exception as e:
                        result_text = f"Error executing tool '{tc.name}': {e}\n{traceback.format_exc()}"
                        is_error = True

                yield ReActToolResultEvent(
                    iteration=iteration,
                    tool_call=tc,
                    result=result_text,
                    is_error=is_error,
                )

                # 把工具结果追加进历史
                context.messages.append(ToolResultMessage(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=[TextContent(text=result_text)],
                    is_error=is_error,
                ))

        # ── 超出最大迭代次数 ──────────────────────────────────────────────
        yield ReActErrorEvent(
            iteration=self._max_iterations,
            reason=f"Exceeded max iterations ({self._max_iterations})",
        )
