"""
Core data types for the provider layer.
Mirrors pi/packages/ai/src/types.ts
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Union


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelCost:
    input: float        # $/million tokens
    output: float
    cache_read: float
    cache_write: float


@dataclass
class Model:
    id: str
    name: str
    api: str            # "openai-completions" | "anthropic-messages" | ...
    provider: str       # "openai" | "anthropic" | "deepseek" | ...
    base_url: str
    reasoning: bool
    input: list[Literal["text", "image"]]
    cost: ModelCost
    context_window: int
    max_tokens: int
    headers: dict[str, str] = field(default_factory=dict)
    # Compatibility overrides (thinkingFormat, supportsStore, etc.)
    compat: dict[str, Any] = field(default_factory=dict)
    # Maps thinking level names to provider-specific values; None = unsupported
    thinking_level_map: dict[str, str | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

@dataclass
class TextContent:
    type: Literal["text"] = "text"
    text: str = ""
    text_signature: str | None = None


@dataclass
class ThinkingContent:
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: str | None = None
    redacted: bool = False


@dataclass
class ImageContent:
    type: Literal["image"] = "image"
    data: str = ""       # base64 encoded
    mime_type: str = ""  # "image/jpeg" | "image/png" | ...


@dataclass
class ToolCall:
    type: Literal["toolCall"] = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    thought_signature: str | None = None  # Google-specific


# ---------------------------------------------------------------------------
# Usage & cost
# ---------------------------------------------------------------------------

@dataclass
class UsageCost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    reasoning: int | None = None
    cost: UsageCost = field(default_factory=UsageCost)


def empty_usage() -> Usage:
    return Usage()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]

AssistantContent = Union[TextContent, ThinkingContent, ToolCall]
UserContentBlock = Union[TextContent, ImageContent]


@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: str | list[UserContentBlock] = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    response_model: str | None = None
    response_id: str | None = None
    usage: Usage = field(default_factory=empty_usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class ToolResultMessage:
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[UserContentBlock] = field(default_factory=list)
    is_error: bool = False
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class Context:
    messages: list[Message]
    system_prompt: str | None = None
    tools: list[Tool] | None = None


# ---------------------------------------------------------------------------
# Stream options
# ---------------------------------------------------------------------------

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
ModelThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


@dataclass
class StreamOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    cache_retention: Literal["none", "short", "long"] = "short"
    session_id: str | None = None
    headers: dict[str, str | None] | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    metadata: dict[str, Any] | None = None
    env: dict[str, str] | None = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    reasoning: ThinkingLevel | None = None


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------

@dataclass
class StartEvent:
    type: Literal["start"] = "start"
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class TextStartEvent:
    type: Literal["text_start"] = "text_start"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class TextDeltaEvent:
    type: Literal["text_delta"] = "text_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class TextEndEvent:
    type: Literal["text_end"] = "text_end"
    content_index: int = 0
    content: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ThinkingStartEvent:
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ThinkingDeltaEvent:
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ThinkingEndEvent:
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int = 0
    content: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ToolCallStartEvent:
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ToolCallDeltaEvent:
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ToolCallEndEvent:
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = 0
    tool_call: ToolCall = field(default_factory=ToolCall)
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class DoneEvent:
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse"] = "stop"
    message: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ErrorEvent:
    type: Literal["error"] = "error"
    reason: Literal["error", "aborted"] = "error"
    error: AssistantMessage = field(default_factory=AssistantMessage)


StreamEvent = Union[
    StartEvent,
    TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
]
