import pytest
import json

from provider.types import (
    UserMessage, AssistantMessage, ToolResultMessage,
    TextContent, ToolCall, ImageContent, Model, ModelCost
)
from provider.api.transform_messages import transform_messages

dummy_model = Model(
    id="test", name="Test", api="test", provider="test", base_url="http",
    reasoning=False, input=["text"], cost=ModelCost(0,0,0,0), context_window=1000, max_tokens=1000
)

def test_transform_simple_user_message():
    msgs = [UserMessage(content="Hello world")]
    result = transform_messages(msgs, dummy_model)
    
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "Hello world"

def test_transform_assistant_message_with_text():
    msgs = [AssistantMessage(content=[TextContent(text="I am assistant")])]
    result = transform_messages(msgs, dummy_model)
    
    assert len(result) == 1
    assert result[0].role == "assistant"
    assert result[0].content[0].type == "text"
    assert result[0].content[0].text == "I am assistant"

def test_transform_assistant_message_with_tool_call():
    tc = ToolCall(id="call_123", name="get_weather", arguments={"loc": "Tokyo"})
    msgs = [
        AssistantMessage(content=[TextContent(text="Checking..."), tc])
    ]
    result = transform_messages(msgs, dummy_model)
    
    # transform_messages synthesizes a ToolResultMessage if one is missing!
    assert len(result) == 2
    assert result[0].role == "assistant"
    assert len(result[0].content) == 2
    assert result[0].content[0].text == "Checking..."
    assert result[0].content[1].type == "toolCall"
    assert result[0].content[1].id == "call_123"
    
    # Synthetic ToolResultMessage
    assert result[1].role == "toolResult"
    assert result[1].tool_call_id == "call_123"
    assert result[1].is_error is True

def test_transform_tool_result_message():
    msgs = [
        ToolResultMessage(
            tool_call_id="call_123",
            tool_name="get_weather",
            content=[TextContent(text="Sunny, 25C")]
        )
    ]
    result = transform_messages(msgs, dummy_model)
    
    assert len(result) == 1
    assert result[0].role == "toolResult"
    assert result[0].tool_call_id == "call_123"
    assert result[0].content[0].text == "Sunny, 25C"

def test_transform_user_message_with_image():
    msgs = [
        UserMessage(content=[
            TextContent(text="What is this?"),
            ImageContent(data="base64data", mime_type="image/jpeg")
        ])
    ]
    result = transform_messages(msgs, dummy_model)
    
    assert len(result) == 1
    assert result[0].role == "user"
    assert isinstance(result[0].content, list)
    assert len(result[0].content) == 2
    
    # If the model does not support images ("image" not in input), it replaces them!
    # dummy_model has input=["text"], so the image is downgraded!
    assert result[0].content[0].type == "text"
    assert result[0].content[0].text == "What is this?"
    assert result[0].content[1].type == "text"
    assert "does not support images" in result[0].content[1].text
