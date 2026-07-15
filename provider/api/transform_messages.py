"""
Message transformation pipeline.
Mirrors pi/packages/ai/src/api/transform-messages.ts
"""
from __future__ import annotations

import copy
from typing import Callable

from ..types import (
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"


def _replace_images_with_placeholder(content: list, placeholder: str) -> list:
    result = []
    previous_was_placeholder = False

    for block in content:
        if getattr(block, "type", None) == "image":
            if not previous_was_placeholder:
                result.append(TextContent(text=placeholder))
            previous_was_placeholder = True
            continue

        result.append(block)
        previous_was_placeholder = (getattr(block, "text", None) == placeholder)

    return result


def _downgrade_unsupported_images(messages: list[Message], model: Model) -> list[Message]:
    if "image" in model.input:
        return messages

    result = []
    for msg in messages:
        if isinstance(msg, UserMessage) and isinstance(msg.content, list):
            new_msg = copy.copy(msg)
            new_msg.content = _replace_images_with_placeholder(msg.content, NON_VISION_USER_IMAGE_PLACEHOLDER)
            result.append(new_msg)
        elif isinstance(msg, ToolResultMessage):
            new_msg = copy.copy(msg)
            new_msg.content = _replace_images_with_placeholder(msg.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER)
            result.append(new_msg)
        else:
            result.append(msg)
            
    return result


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None = None
) -> list[Message]:
    """
    Transforms messages for API requests:
    1. Downgrades unsupported images
    2. Filters out encrypted/redacted thinking blocks for cross-model
    3. Normalizes tool call IDs
    4. Synthesizes empty tool results for orphaned tool calls
    """
    tool_call_id_map: dict[str, str] = {}
    
    image_aware_messages = _downgrade_unsupported_images(messages, model)
    transformed: list[Message] = []
    
    # First pass: transform messages
    for msg in image_aware_messages:
        if isinstance(msg, UserMessage):
            transformed.append(msg)
            
        elif isinstance(msg, ToolResultMessage):
            normalized_id = tool_call_id_map.get(msg.tool_call_id)
            if normalized_id and normalized_id != msg.tool_call_id:
                new_msg = copy.copy(msg)
                new_msg.tool_call_id = normalized_id
                transformed.append(new_msg)
            else:
                transformed.append(msg)
                
        elif isinstance(msg, AssistantMessage):
            is_same_model = (
                msg.provider == model.provider and
                msg.api == model.api and
                msg.model == model.id
            )
            
            transformed_content = []
            for block in msg.content:
                if isinstance(block, ThinkingContent):
                    if block.redacted:
                        if is_same_model:
                            transformed_content.append(block)
                        continue
                        
                    if is_same_model and block.thinking_signature:
                        transformed_content.append(block)
                        continue
                        
                    if not block.thinking or not block.thinking.strip():
                        continue
                        
                    if is_same_model:
                        transformed_content.append(block)
                    else:
                        transformed_content.append(TextContent(text=block.thinking))
                        
                elif isinstance(block, TextContent):
                    if is_same_model:
                        transformed_content.append(block)
                    else:
                        transformed_content.append(TextContent(text=block.text))
                        
                elif isinstance(block, ToolCall):
                    normalized_tool_call = copy.copy(block)
                    if not is_same_model and normalized_tool_call.thought_signature:
                        normalized_tool_call.thought_signature = None
                        
                    if not is_same_model and normalize_tool_call_id:
                        normalized_id = normalize_tool_call_id(block.id, model, msg)
                        if normalized_id != block.id:
                            tool_call_id_map[block.id] = normalized_id
                            normalized_tool_call.id = normalized_id
                            
                    transformed_content.append(normalized_tool_call)
                else:
                    transformed_content.append(block)
                    
            new_msg = copy.copy(msg)
            new_msg.content = transformed_content
            transformed.append(new_msg)
            
    # Second pass: synthesize tool results
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()
    
    def insert_synthetic_tool_results():
        nonlocal pending_tool_calls, existing_tool_result_ids
        if pending_tool_calls:
            for tc in pending_tool_calls:
                if tc.id not in existing_tool_result_ids:
                    result.append(ToolResultMessage(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[TextContent(text="No result provided")],
                        is_error=True
                    ))
            pending_tool_calls = []
            existing_tool_result_ids = set()

    for msg in transformed:
        if isinstance(msg, AssistantMessage):
            insert_synthetic_tool_results()
            if msg.stop_reason in ("error", "aborted"):
                continue
                
            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()
            result.append(msg)
            
        elif isinstance(msg, ToolResultMessage):
            existing_tool_result_ids.add(msg.tool_call_id)
            result.append(msg)
            
        elif isinstance(msg, UserMessage):
            insert_synthetic_tool_results()
            result.append(msg)
            
    insert_synthetic_tool_results()
    return result


'''
transform_messages.py 并不是专门绑定给 openai_completions.py 的，它是一个独立于底层 API 协议的“通用消息中间件（Middleware）”。

在我们的架构中，消息流转是分两步走的：

第一步：逻辑归一化 (transform_messages.py)
它的输入是：我们的标准 Message 数组。 它的输出是：依旧是标准 Message 数组，但是经过了重新编排。 它的职责是处理不同大模型在逻辑能力上的残缺或怪癖。比如：

OpenAI 的 o1 模型不支持 system 角色，transform_messages.py 就会把 system 强行塞进第一个 user 消息里。
某些老旧的开源模型不支持连续发两条 user 消息，它就会把多条 user 消息合并成一条。
这部分逻辑只和 compat（模型的特性档案）有关，和走什么网络协议完全无关。
第二步：协议序列化 (比如 openai_completions.py)
它的输入是：刚才归一化好的标准 Message 数组。 它的输出是：发往网络的 HTTP JSON 参数。 它的职责仅仅是将标准结构翻译成特定厂商的 API 格式。

openai_completions.py 负责把它翻译成 [{"role": "user", "content": "..."}]。
如果未来我们接入了 Anthropic (Claude)，我们会写一个 anthropic_messages.py，它负责把同样的 Message 翻译成 Claude 特有的结构。
总结： transform_messages.py 是一道“安检门”。不管你是发往 OpenAI 还是发往未来的 Claude，所有消息在发送前，都要先过这道门，把模型不认识的语法（比如孤儿 ToolCall、不支持的 system 提示词）统统在这个阶段清洗干净！
'''