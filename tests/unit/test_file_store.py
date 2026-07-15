import os
import json
import pytest
from pathlib import Path
from provider.auth.file_store import FileCredentialStore
from provider.auth.types import ApiKeyCredential, OAuthCredential

@pytest.fixture
def temp_credentials_file(tmp_path):
    return tmp_path / ".credentials.json"

@pytest.mark.asyncio
async def test_file_credential_store_read_write(temp_credentials_file):
    store = FileCredentialStore(temp_credentials_file)
    
    # Initially empty
    assert await store.read("openai") is None
    
    # Write api key
    cred = ApiKeyCredential(key="test-key")
    await store.write("openai", cred)
    
    # Read back
    stored = await store.read("openai")
    assert isinstance(stored, ApiKeyCredential)
    assert stored.key == "test-key"
    
    # Check file contents
    with open(temp_credentials_file, "r") as f:
        data = json.load(f)
        assert data["openai"]["type"] == "api_key"
        assert data["openai"]["key"] == "test-key"

@pytest.mark.asyncio
async def test_file_credential_store_delete(temp_credentials_file):
    store = FileCredentialStore(temp_credentials_file)
    await store.write("openai", ApiKeyCredential(key="test"))
    assert await store.read("openai") is not None
    
    await store.delete("openai")
    assert await store.read("openai") is None
