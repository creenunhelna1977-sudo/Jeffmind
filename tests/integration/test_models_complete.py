"""
integration/test_models_complete.py
测试 Models.complete 将流事件聚合成最终 AssistantMessage 的逻辑。
同时测试 Models.stream 经过 auth resolve 后的完整链路。
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from provider.models import Models, create_provider
from provider.auth.types import ProviderAuth, ModelAuth, AuthResult, ApiKeyAuth, ApiKeyCredential
from provider.auth.store import InMemoryCredentialStore
from provider.types import (
    Model, ModelCost, Context, UserMessage, StreamOptions,
    AssistantMessage, TextContent, ToolCall,
    StartEvent, TextDeltaEvent, DoneEvent, ErrorEvent
)
from provider.api.openai_completions import openai_completions_api


# ── 工具函数 ──────────────────────────────────────────────────────────────

class StaticApiKeyAuth(ApiKeyAuth):
    """测试用：永远返回固定 API key。"""
    name = "static"

    def __init__(self, key: str):
        self._key = key

    async def resolve(self, model, context, credential=None):
        return AuthResult(auth=ModelAuth(api_key=self._key), source="static")


def make_test_model(provider_id="test") -> Model:
    return Model(
        id="test-model", name="Test", api="openai-completions",
        provider=provider_id, base_url="http://test",
        reasoning=False, input=["text"],
        cost=ModelCost(0, 0, 0, 0),
        context_window=1000, max_tokens=1000
    )


def make_chunk(content=None, finish_reason=None, tool_calls=None, chunk_id="1"):
    chunk = MagicMock()
    chunk.id = chunk_id
    choice = MagicMock()
    choice.finish_reason = finish_reason
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice.delta = delta
    chunk.choices = [choice]
    chunk.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15) if finish_reason else None
    return chunk


def make_models_with_mock_stream(chunks: list) -> tuple[Models, Model]:
    """Helper：创建一个注入了 mock 流的 Models 注册表。"""
    model = make_test_model()
    api = openai_completions_api()
    provider = create_provider(
        id="test", name="Test", base_url="http://test",
        auth=ProviderAuth(api_key=StaticApiKeyAuth("test-key")),
        models=[model], api=api
    )
    models = Models()
    models.set_provider(provider)
    return models, model, chunks


# ── 测试 Models.complete ──────────────────────────────────────────────────

class TestModelsComplete:
    async def test_complete_returns_assistant_message_with_text(self):
        """complete() 聚合所有 TextDeltaEvent，返回完整 AssistantMessage。"""
        model = make_test_model()
        chunks = [
            make_chunk(content="Hello", chunk_id="1"),
            make_chunk(content=" World", chunk_id="2"),
            make_chunk(finish_reason="stop", chunk_id="3"),
        ]

        async def fake_stream():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = fake_stream()

        with patch("provider.api.openai_completions.AsyncOpenAI", return_value=mock_client):
            api = openai_completions_api()
            provider = create_provider(
                id="test", name="Test", base_url="http://test",
                auth=ProviderAuth(api_key=StaticApiKeyAuth("test-key")),
                models=[model], api=api
            )
            registry = Models()
            registry.set_provider(provider)

            ctx = Context(messages=[UserMessage(content="Hi")])
            message = await registry.complete(model, ctx)

        assert isinstance(message, AssistantMessage)
        assert message.stop_reason == "stop"
        assert len(message.content) == 1
        assert message.content[0].text == "Hello World"
        assert message.usage.total_tokens == 15

    async def test_complete_returns_error_message_on_failure(self):
        """complete() 发生错误时返回 stop_reason='error' 的 AssistantMessage。"""
        model = make_test_model()
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection timeout")

        with patch("provider.api.openai_completions.AsyncOpenAI", return_value=mock_client):
            api = openai_completions_api()
            provider = create_provider(
                id="test", name="Test", base_url="http://test",
                auth=ProviderAuth(api_key=StaticApiKeyAuth("test-key")),
                models=[model], api=api
            )
            registry = Models()
            registry.set_provider(provider)
            ctx = Context(messages=[UserMessage(content="Hi")])
            message = await registry.complete(model, ctx)

        assert message.stop_reason == "error"
        assert "Connection timeout" in message.error_message

    async def test_complete_with_tool_calls_aggregates_correctly(self):
        """complete() 能正确聚合工具调用流，arguments 被正确解析为 dict。"""
        model = make_test_model()

        def make_tc_delta(index, id=None, name=None, args=None):
            tc = MagicMock()
            tc.index = index
            tc.id = id
            tc.function = MagicMock()
            tc.function.name = name
            tc.function.arguments = args
            return tc

        chunks = [
            make_chunk(tool_calls=[make_tc_delta(0, id="call_xyz", name="get_weather", args='{"city":')], chunk_id="1"),
            make_chunk(tool_calls=[make_tc_delta(0, args='"Tokyo"}')], chunk_id="2"),
            make_chunk(finish_reason="tool_calls", chunk_id="3"),
        ]

        async def fake_stream():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = fake_stream()

        with patch("provider.api.openai_completions.AsyncOpenAI", return_value=mock_client):
            api = openai_completions_api()
            provider = create_provider(
                id="test", name="Test", base_url="http://test",
                auth=ProviderAuth(api_key=StaticApiKeyAuth("test-key")),
                models=[model], api=api
            )
            registry = Models()
            registry.set_provider(provider)
            ctx = Context(messages=[UserMessage(content="What's the weather in Tokyo?")])
            message = await registry.complete(model, ctx)

        assert message.stop_reason == "toolUse"
        tool_calls = [b for b in message.content if isinstance(b, ToolCall)]
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        # 验证 arguments 已被正确解析为 dict
        assert tool_calls[0].arguments == {"city": "Tokyo"}


# ── 测试 Models.stream 通过 provider 不存在 ──────────────────────────────

class TestModelsStreamErrors:
    async def test_stream_raises_when_provider_not_found(self):
        """请求不存在的 provider 时抛出 ModelsError。"""
        from provider.models import ModelsError
        registry = Models()
        ghost_model = make_test_model("ghost-provider")
        ctx = Context(messages=[UserMessage(content="Hi")])
        with pytest.raises(ModelsError, match="Provider not found"):
            async for _ in registry.stream(ghost_model, ctx):
                pass

    async def test_auth_key_injected_into_options(self):
        """鉴权 API key 被正确注入到最终请求选项中。"""
        model = make_test_model()
        captured_options = {}

        async def fake_stream_impl(model, ctx, options):
            captured_options["api_key"] = options.api_key
            # 立刻结束，不 yield 任何事件
            return
            yield  # 让它成为 generator

        api = openai_completions_api()
        original_stream = api.stream

        async def patched_stream(m, c, o):
            captured_options["api_key"] = o.api_key
            # yield 一个最简单的 done event 让流正常结束
            from provider.types import DoneEvent
            yield DoneEvent(reason="stop", message=AssistantMessage())

        api.stream = patched_stream

        provider = create_provider(
            id="test", name="Test", base_url="http://test",
            auth=ProviderAuth(api_key=StaticApiKeyAuth("injected-key")),
            models=[model], api=api
        )
        registry = Models()
        registry.set_provider(provider)
        ctx = Context(messages=[UserMessage(content="Hi")])

        async for _ in registry.stream(model, ctx):
            pass

        assert captured_options.get("api_key") == "injected-key"
