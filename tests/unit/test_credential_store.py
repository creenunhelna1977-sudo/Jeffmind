"""
unit/test_credential_store.py
测试 InMemoryCredentialStore 的 CRUD 操作和并发安全性。
"""
import asyncio
import pytest
from provider.auth.store import InMemoryCredentialStore
from provider.auth.types import ApiKeyCredential, OAuthCredential


@pytest.fixture
def store():
    return InMemoryCredentialStore()


class TestInMemoryCredentialStore:
    async def test_read_empty_returns_none(self, store):
        result = await store.read("openai")
        assert result is None

    async def test_write_and_read_api_key(self, store):
        cred = ApiKeyCredential(key="sk-test-123")
        await store.write("openai", cred)
        result = await store.read("openai")
        assert result is not None
        assert result.type == "api_key"
        assert result.key == "sk-test-123"

    async def test_write_and_read_oauth(self, store):
        cred = OAuthCredential(access_token="token-abc", refresh_token="refresh-xyz", expires=9999999999000)
        await store.write("github", cred)
        result = await store.read("github")
        assert result is not None
        assert result.type == "oauth"
        assert result.access_token == "token-abc"

    async def test_overwrite_credential(self, store):
        cred1 = ApiKeyCredential(key="old-key")
        cred2 = ApiKeyCredential(key="new-key")
        await store.write("openai", cred1)
        await store.write("openai", cred2)
        result = await store.read("openai")
        assert result.key == "new-key"

    async def test_delete_credential(self, store):
        cred = ApiKeyCredential(key="sk-test")
        await store.write("openai", cred)
        await store.delete("openai")
        result = await store.read("openai")
        assert result is None

    async def test_delete_nonexistent_is_safe(self, store):
        # 删除不存在的 key 不应抛出异常
        await store.delete("nonexistent")

    async def test_multiple_providers_isolated(self, store):
        await store.write("openai", ApiKeyCredential(key="openai-key"))
        await store.write("deepseek", ApiKeyCredential(key="deepseek-key"))
        openai_cred = await store.read("openai")
        deepseek_cred = await store.read("deepseek")
        assert openai_cred.key == "openai-key"
        assert deepseek_cred.key == "deepseek-key"

    async def test_concurrent_writes_are_safe(self, store):
        """并发写入相同 provider 不应导致 race condition 或数据损坏。"""
        async def write_key(key: str):
            await store.write("openai", ApiKeyCredential(key=key))

        # 同时发起 10 个写入
        await asyncio.gather(*[write_key(f"key-{i}") for i in range(10)])

        # 结果应是某一次写入的有效值
        result = await store.read("openai")
        assert result is not None
        assert result.key.startswith("key-")
