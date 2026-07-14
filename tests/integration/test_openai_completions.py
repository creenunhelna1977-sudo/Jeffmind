import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from provider.api.openai_completions import openai_completions_api
from provider.types import (
    Model, ModelCost, Context, UserMessage, 
    StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent, DoneEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ErrorEvent
)

@pytest.fixture
def mock_model():
    return Model(
        id="test-model",
        name="Test",
        api="openai-completions",
        provider="test-provider",
        base_url="http://test",
        reasoning=False,
        input=["text"],
        cost=ModelCost(0, 0, 0, 0),
        context_window=1000,
        max_tokens=1000
    )

@pytest.fixture
def mock_context():
    return Context(messages=[UserMessage(content="Hi")])

# Helper to create mock streaming chunks
def create_mock_chunk(content_text=None, tool_calls=None, finish_reason=None, chunk_id="chunk-1"):
    chunk = MagicMock()
    chunk.id = chunk_id
    
    choice = MagicMock()
    choice.finish_reason = finish_reason
    
    delta = MagicMock()
    delta.content = content_text
    delta.tool_calls = tool_calls
    
    choice.delta = delta
    chunk.choices = [choice]
    
    # Add dummy usage for the last chunk
    chunk.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30) if finish_reason else None
    
    return chunk

@pytest.mark.asyncio
async def test_openai_stream_plain_text(mock_model, mock_context):
    api = openai_completions_api()
    
    # 1. Setup mock AsyncOpenAI client
    mock_client = AsyncMock()
    mock_stream = AsyncMock()
    
    # Define the sequence of chunks the stream will yield
    chunks = [
        create_mock_chunk(content_text="Hello", chunk_id="1"),
        create_mock_chunk(content_text=" world", chunk_id="2"),
        create_mock_chunk(finish_reason="stop", chunk_id="3")
    ]
    
    async def async_generator():
        for chunk in chunks:
            yield chunk
            
    mock_client.chat.completions.create.return_value = async_generator()
    
    # 2. Inject mock client
    with patch("provider.api.openai_completions.AsyncOpenAI", return_value=mock_client):
        from provider.types import StreamOptions
        options = StreamOptions(api_key="test-key", max_tokens=100)
        
        events = []
        async for event in api.stream(mock_model, mock_context, options):
            events.append(event)
            
        # 3. Assert correct event sequence
        assert len(events) == 4
        assert isinstance(events[0], StartEvent)
        assert isinstance(events[1], TextDeltaEvent)
        assert events[1].delta == "Hello"
        assert isinstance(events[2], TextDeltaEvent)
        assert events[2].delta == " world"
        assert isinstance(events[3], DoneEvent)
        
        # Verify AssistantMessage was constructed correctly
        msg = events[3].message
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"
        assert msg.content[0].text == "Hello world"
        assert msg.usage.total_tokens == 30

@pytest.mark.asyncio
async def test_openai_stream_tool_calls(mock_model, mock_context):
    api = openai_completions_api()
    
    # 1. Setup mock AsyncOpenAI client
    mock_client = AsyncMock()
    mock_stream = AsyncMock()
    
    # Create mock tool call deltas
    def create_tc_delta(index, id=None, name=None, arguments=None):
        tc = MagicMock()
        tc.index = index
        tc.id = id
        tc.function = MagicMock()
        tc.function.name = name
        tc.function.arguments = arguments
        return tc
    
    chunks = [
        create_mock_chunk(tool_calls=[create_tc_delta(0, id="call_1", name="get_weather", arguments="{\n")], chunk_id="1"),
        create_mock_chunk(tool_calls=[create_tc_delta(0, arguments='  "loc": "Tokyo"\n}')], chunk_id="2"),
        create_mock_chunk(finish_reason="tool_calls", chunk_id="3")
    ]
    
    async def async_generator():
        for chunk in chunks:
            yield chunk
            
    mock_client.chat.completions.create.return_value = async_generator()
    
    # 2. Inject mock client
    with patch("provider.api.openai_completions.AsyncOpenAI", return_value=mock_client):
        from provider.types import StreamOptions
        options = StreamOptions(api_key="test-key")
        
        events = []
        async for event in api.stream(mock_model, mock_context, options):
            events.append(event)
            
        # 3. Assert correct event sequence
        # Expect: Start, ToolCallStart, ToolCallDelta, ToolCallDelta, ToolCallEnd, Done
        assert len(events) == 6
        assert isinstance(events[0], StartEvent)
        assert isinstance(events[1], ToolCallStartEvent)
        assert events[1].partial.content[0].name == "get_weather"
        
        assert isinstance(events[2], ToolCallDeltaEvent)
        assert events[2].delta == "{\n"
        
        assert isinstance(events[3], ToolCallDeltaEvent)
        assert events[3].delta == '  "loc": "Tokyo"\n}'
        
        from provider.types import ToolCallEndEvent as TCEndEvent
        assert isinstance(events[4], TCEndEvent)
        assert events[4].tool_call.name == "get_weather"
        
        assert isinstance(events[5], DoneEvent)
        
        msg = events[5].message
        assert len(msg.content) == 1
        assert msg.content[0].type == "toolCall"
        
        # Verify that buffer parsing logic worked!
        assert msg.content[0].arguments == {"loc": "Tokyo"}

@pytest.mark.asyncio
async def test_openai_stream_api_error(mock_model, mock_context):
    api = openai_completions_api()
    
    mock_client = AsyncMock()
    # Simulate a 404 error
    mock_client.chat.completions.create.side_effect = Exception("Error code: 404 - Not Found")
    
    with patch("provider.api.openai_completions.AsyncOpenAI", return_value=mock_client):
        from provider.types import StreamOptions
        options = StreamOptions(api_key="test-key")
            
        events = []
        async for event in api.stream(mock_model, mock_context, options):
            events.append(event)
            
        assert len(events) == 2
        assert isinstance(events[0], StartEvent)
        assert isinstance(events[1], ErrorEvent)
        assert "404" in events[1].error.error_message
