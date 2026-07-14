import pytest
import os
from unittest.mock import patch


from provider.auth.helpers import env_api_key_auth
from provider.auth.types import ProviderAuth, ModelAuth
from provider.types import Model, Context, ModelCost

@pytest.fixture
def mock_model():
    return Model(
        id="test-model",
        name="Test",
        api="test",
        provider="test-provider",
        base_url="http://test",
        reasoning=False,
        input=["text"],
        cost=ModelCost(0, 0, 0, 0),
        context_window=1000,
        max_tokens=1000
    )

@pytest.mark.asyncio
async def test_env_api_key_auth_found(mock_model):
    auth = env_api_key_auth("Test Key", ["TEST_API_KEY"])
    context = Context(messages=[])
    
    with patch.dict(os.environ, {"TEST_API_KEY": "secret-123"}):
        result = await auth.resolve(mock_model, context)
        
    assert result is not None
    assert isinstance(result.auth, ModelAuth)
    assert result.auth.api_key == "secret-123"
    assert result.source == "TEST_API_KEY"

@pytest.mark.asyncio
async def test_env_api_key_auth_not_found(mock_model):
    auth = env_api_key_auth("Test Key", ["NON_EXISTENT_KEY"])
    context = Context(messages=[])
    
    with patch.dict(os.environ, {}, clear=True):
        result = await auth.resolve(mock_model, context)
        
    assert result is None
