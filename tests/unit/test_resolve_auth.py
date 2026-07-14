"""
unit/test_resolve_auth.py
测试 resolve_provider_auth 鉴权优先级链。
优先级：options.api_key > 已存凭证 > 环境变量
"""
import os
import pytest
from unittest.mock import patch

from provider.auth.store import InMemoryCredentialStore
from provider.auth.types import ApiKeyCredential, AuthResult, ModelAuth, ProviderAuth
from provider.auth.helpers import env_api_key_auth
from provider.auth.resolve import resolve_provider_auth
from provider.models import create_provider
from provider.types import Model, ModelCost, Context, StreamOptions


def make_provider(env_var: str = "TEST_API_KEY"):
    auth = env_api_key_auth("Test", [env_var])
    dummy_model = Model(
        id="m", name="M", api="test", provider="test-provider", base_url="http://test",
        reasoning=False, input=["text"], cost=ModelCost(0, 0, 0, 0),
        context_window=1000, max_tokens=1000
    )
    provider = create_provider(
        id="test-provider", name="Test", base_url="http://test",
        auth=ProviderAuth(api_key=auth),
        models=[dummy_model], api=None
    )
    return provider, dummy_model


class TestResolveProviderAuth:
    async def test_options_api_key_takes_highest_priority(self):
        """options.api_key 优先级最高，覆盖一切其他来源。"""
        provider, model = make_provider()
        store = InMemoryCredentialStore()
        # 即便环境变量存在
        with patch.dict(os.environ, {"TEST_API_KEY": "env-key"}):
            opts = StreamOptions(api_key="override-key")
            result = await resolve_provider_auth(provider, model, Context(messages=[]), opts, store)
        assert result is not None
        assert result.auth.api_key == "override-key"
        assert result.source == "options.apiKey"

    async def test_stored_credential_used_when_no_options(self):
        """没有 options.api_key 时，使用已存储的 ApiKeyCredential。"""
        provider, model = make_provider()
        store = InMemoryCredentialStore()
        await store.write("test-provider", ApiKeyCredential(key="stored-key"))
        result = await resolve_provider_auth(provider, model, Context(messages=[]), None, store)
        assert result is not None
        assert result.auth.api_key == "stored-key"

    async def test_env_variable_fallback(self):
        """没有 options，没有存储凭证时，回落到环境变量。"""
        provider, model = make_provider("MY_API_KEY")
        store = InMemoryCredentialStore()
        with patch.dict(os.environ, {"MY_API_KEY": "env-key-xyz"}):
            result = await resolve_provider_auth(provider, model, Context(messages=[]), None, store)
        assert result is not None
        assert result.auth.api_key == "env-key-xyz"
        assert result.source == "MY_API_KEY"

    async def test_returns_none_when_no_auth_found(self):
        """所有来源均无鉴权信息时返回 None。"""
        provider, model = make_provider("NONEXISTENT_KEY")
        store = InMemoryCredentialStore()
        with patch.dict(os.environ, {}, clear=True):
            result = await resolve_provider_auth(provider, model, Context(messages=[]), None, store)
        assert result is None

    async def test_stored_credential_overrides_env(self):
        """存储凭证优先级高于环境变量。"""
        provider, model = make_provider("TEST_API_KEY")
        store = InMemoryCredentialStore()
        await store.write("test-provider", ApiKeyCredential(key="stored-wins"))
        with patch.dict(os.environ, {"TEST_API_KEY": "env-key"}):
            result = await resolve_provider_auth(provider, model, Context(messages=[]), None, store)
        assert result.auth.api_key == "stored-wins"
