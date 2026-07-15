"""
OpenAI Completions API implementation.
Mirrors pi/packages/ai/src/api/openai-completions.ts
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from ..models import ProviderStreams
from ..types import (
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from .transform_messages import transform_messages


class OpenAICompatCacheControl:
    pass


def _detect_compat(model: Model) -> dict[str, Any]:
    """Auto-detects compatibility settings from provider name and baseUrl."""
    provider = model.provider
    base_url = model.base_url

    is_zai = provider in ("zai", "zai-coding-cn") or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cf = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cf_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url

    is_non_standard = any([
        is_nvidia, provider == "cerebras", "cerebras.ai" in base_url,
        provider == "xai", "api.x.ai" in base_url, is_together,
        "chutes.ai" in base_url, "deepseek.com" in base_url, is_zai,
        is_moonshot, provider == "opencode", "opencode.ai" in base_url,
        is_cf, is_cf_gateway, is_ant_ling
    ])

    use_max_tokens = any([
        "chutes.ai" in base_url, is_moonshot, is_cf_gateway, is_together, is_nvidia, is_ant_ling
    ])

    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url
    
    thinking_format = "openai"
    if is_deepseek: thinking_format = "deepseek"
    elif is_zai: thinking_format = "zai"
    elif is_together: thinking_format = "together"
    elif is_ant_ling: thinking_format = "ant-ling"
    elif is_openrouter: thinking_format = "openrouter"

    return {
        "supports_store": not is_non_standard,
        "supports_developer_role": not is_non_standard and not is_openrouter,
        "supports_reasoning_effort": not any([is_grok, is_zai, is_moonshot, is_together, is_cf_gateway, is_nvidia, is_ant_ling]),
        "supports_usage_in_streaming": True,
        "max_tokens_field": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requires_tool_result_name": False,
        "requires_assistant_after_tool_result": False,
        "requires_thinking_as_text": False,
        "requires_reasoning_content_on_assistant_messages": is_deepseek,
        "thinking_format": thinking_format,
        "supports_strict_mode": not any([is_moonshot, is_together, is_cf_gateway, is_nvidia]),
        "supports_long_cache_retention": not any([is_together, is_cf, is_cf_gateway, is_nvidia, is_ant_ling]),
    }


def _get_compat(model: Model) -> dict[str, Any]:
    detected = _detect_compat(model)
    compat = dict(detected)
    if model.compat:
        for k, v in model.compat.items():
            compat[k] = v
    return compat


def _normalize_tool_call_id(id_str: str, model: Model, source: AssistantMessage) -> str:
    if "|" in id_str:
        call_id = id_str.split("|")[0]
        # Very simple sanitize: just take first 40 chars
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in call_id)[:40]
    if model.provider == "openai":
        return id_str[:40] if len(id_str) > 40 else id_str
    return id_str


def _convert_messages(model: Model, context: Context, compat: dict[str, Any]) -> list[ChatCompletionMessageParam]:
    transformed = transform_messages(context.messages, model, _normalize_tool_call_id)
    params: list[ChatCompletionMessageParam] = []

    if context.system_prompt:
        role = "developer" if model.reasoning and compat["supports_developer_role"] else "system"
        # We type ignore because python SDK might not have 'developer' literally typed in older versions
        params.append({"role": role, "content": context.system_prompt}) # type: ignore

    last_role = None
    for msg in transformed:
        if msg.role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": msg.content})
            else:
                # Need to handle complex content (images)
                content_parts = []
                for part in msg.content:
                    if getattr(part, "type", None) == "text":
                        content_parts.append({"type": "text", "text": getattr(part, "text", "")})
                    elif getattr(part, "type", None) == "image":
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{getattr(part, 'mime_type')};base64,{getattr(part, 'data')}"}
                        })
                params.append({"role": "user", "content": content_parts})
                
        elif msg.role == "assistant":
            # Just simple text concatenation for now
            text_parts = [b.text for b in msg.content if isinstance(b, TextContent)]
            assistant_text = "".join(text_parts)
            
            tool_calls_blocks = [b for b in msg.content if isinstance(b, ToolCall)]
            
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text if assistant_text else None
            }
            
            if tool_calls_blocks:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    } for tc in tool_calls_blocks
                ]
            
            # Note: skipping reasoning content serialization for brevity unless requested.
            
            params.append(assistant_msg) # type: ignore
            
        elif msg.role == "toolResult":
            # OpenAI represents tool results as 'tool' role messages
            params.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": "".join([getattr(c, "text", "") for c in msg.content])
            })
            
        last_role = msg.role

    return params


def _convert_tools(tools: list, compat: dict[str, Any]) -> list[ChatCompletionToolParam]:
    res = []
    for t in tools:
        tool_param: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
        }
        if compat.get("supports_strict_mode", True):
            tool_param["function"]["strict"] = False
        res.append(tool_param) # type: ignore
    return res


def _build_params(model: Model, context: Context, options: StreamOptions | None, compat: dict[str, Any]) -> dict[str, Any]:
    messages = _convert_messages(model, context, compat)
    
    params: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }
    
    if compat.get("supports_usage_in_streaming", True):
        params["stream_options"] = {"include_usage": True}
        
    if compat.get("supports_store", True):
        params["store"] = False
        
    if options and options.max_tokens:
        params[compat["max_tokens_field"]] = options.max_tokens
        
    if options and options.temperature is not None:
        params["temperature"] = options.temperature
        
    if context.tools:
        params["tools"] = _convert_tools(context.tools, compat)
        
    reasoning_effort = getattr(options, "reasoning", None) if options else None
    
    if compat["thinking_format"] == "deepseek" and model.reasoning:
        if reasoning_effort:
            params["thinking"] = {"type": "enabled"}
            
    return params


class OpenAICompletionsApi(ProviderStreams):
    async def _do_stream(self, model: Model, context: Context, options: StreamOptions | None) -> AsyncGenerator[StreamEvent, None]:
        from ..types import StartEvent, DoneEvent, ErrorEvent, TextDeltaEvent, Usage, AssistantMessage
        
        output = AssistantMessage(
            role="assistant",
            api=model.api,
            provider=model.provider,
            model=model.id,
            content=[],
        )
        
        try:
            api_key = options.api_key if options else None
            compat = _get_compat(model)
            
            headers = dict(model.headers)
            if options and options.headers:
                headers.update(options.headers)
                
            client = AsyncOpenAI(
                api_key=api_key or "dummy",
                base_url=model.base_url,
                default_headers=headers,
            )
            
            params = _build_params(model, context, options, compat)
            
            yield StartEvent(partial=output)
            
            stream = await client.chat.completions.create(**params)
            
            text_block = None
            tool_blocks: dict[int, ToolCall] = {}
            tool_args_buffer: dict[int, str] = {}
            
            async for chunk in stream:
                if chunk.id and not output.response_id:
                    output.response_id = chunk.id
                if chunk.model and chunk.model != model.id:
                    output.response_model = chunk.model
                    
                if chunk.usage:
                    u = chunk.usage
                    output.usage = Usage(
                        input=u.prompt_tokens,
                        output=u.completion_tokens,
                        total_tokens=u.total_tokens
                    )
                    
                if not chunk.choices:
                    continue
                    
                choice = chunk.choices[0]
                
                if choice.finish_reason:
                    if choice.finish_reason == "tool_calls":
                        output.stop_reason = "toolUse"
                    elif choice.finish_reason == "length":
                        output.stop_reason = "length"
                    else:
                        output.stop_reason = "stop"
                    
                delta = choice.delta
                if delta.content:
                    if not text_block:
                        text_block = TextContent()
                        output.content.append(text_block)
                    text_block.text += delta.content
                    yield TextDeltaEvent(
                        content_index=output.content.index(text_block),
                        delta=delta.content,
                        partial=output
                    )
                    
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_blocks[tc_delta.index] = ToolCall(
                                    id=tc_delta.id or "",
                                    name=tc_delta.function.name,
                                    arguments={}
                                )
                                tool_args_buffer[tc_delta.index] = ""
                                output.content.append(tool_blocks[tc_delta.index])
                                yield ToolCallStartEvent(
                                    content_index=len(output.content) - 1,
                                    partial=output
                                )
                            
                            if tc_delta.function.arguments:
                                tool_args_buffer[tc_delta.index] += tc_delta.function.arguments
                                yield ToolCallDeltaEvent(
                                    content_index=output.content.index(tool_blocks[tc_delta.index]),
                                    delta=tc_delta.function.arguments,
                                    partial=output
                                )
                            
            # End of stream: parse buffered args and emit ToolCallEndEvent
            for idx, tc in tool_blocks.items():
                if idx in tool_args_buffer:
                    try:
                        import json
                        tc.arguments = json.loads(tool_args_buffer[idx])
                    except Exception:
                        pass
                yield ToolCallEndEvent(
                    content_index=output.content.index(tc),
                    tool_call=tc,
                    partial=output
                )

            yield DoneEvent(reason=output.stop_reason, message=output)
            
        except Exception as e:
            output.stop_reason = "error"
            
            error_msg = str(e)
            # 尝试捕获 openai 的详细 HTTP 报错体
            if hasattr(e, "response") and hasattr(e.response, "text"):
                error_msg += f" - Response: {e.response.text}"
            elif hasattr(e, "body"):
                error_msg += f" - Body: {getattr(e, 'body')}"
                
            output.error_message = error_msg
            yield ErrorEvent(reason="error", error=output)

    async def stream(self, model: Model, context: Context, options: StreamOptions | None = None) -> AsyncGenerator[StreamEvent, None]:
        async for event in self._do_stream(model, context, options):
            yield event

    async def stream_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None) -> AsyncGenerator[StreamEvent, None]:
        async for event in self._do_stream(model, context, options):
            yield event


def openai_completions_api() -> ProviderStreams:
    return OpenAICompletionsApi()
