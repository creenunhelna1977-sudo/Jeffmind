import pytest

from provider.models import Models, create_provider
from provider.auth.types import ProviderAuth, ModelAuth, AuthResult, ApiKeyAuth
from provider.types import Model, ModelCost

class DummyAuth(ApiKeyAuth):
    async def resolve(self, model, context, credential=None):
        return AuthResult(auth=ModelAuth(api_key="dummy"), source="dummy")

@pytest.fixture
def models_registry():
    models = Models()
    
    # Register a dummy provider
    dummy_models = [
        Model(
            id="dummy-1",
            name="Dummy 1",
            api="dummy-api",
            provider="dummy",
            base_url="http://dummy",
            reasoning=False,
            input=["text"],
            cost=ModelCost(0,0,0,0),
            context_window=1000,
            max_tokens=1000
        )
    ]
    
    provider = create_provider(
        id="dummy",
        name="Dummy",
        base_url="http://dummy",
        auth=ProviderAuth(api_key=DummyAuth()),
        models=dummy_models,
        api=None # Not testing API here
    )
    models.set_provider(provider)
    return models

def test_get_model_found(models_registry):
    model = models_registry.get_model("dummy", "dummy-1")
    assert model is not None
    assert model.id == "dummy-1"
    assert model.provider == "dummy"

def test_get_model_not_found(models_registry):
    model = models_registry.get_model("dummy", "non-existent")
    assert model is None

def test_get_model_provider_not_found(models_registry):
    model = models_registry.get_model("unknown", "dummy-1")
    assert model is None

def test_get_models_list(models_registry):
    models_list = models_registry.get_models("dummy")
    assert len(models_list) == 1
    assert models_list[0].id == "dummy-1"

@pytest.mark.asyncio
async def test_refresh_models():
    models_registry = Models()
    
    async def mock_refresh():
        return [
            Model(
                id="dynamic-1", name="Dynamic 1", api="dummy-api", provider="dynamic", base_url="http",
                reasoning=False, input=["text"], cost=ModelCost(0,0,0,0), context_window=1000, max_tokens=1000
            )
        ]
        
    provider = create_provider(
        id="dynamic",
        name="Dynamic",
        base_url="http",
        auth=ProviderAuth(api_key=DummyAuth()),
        models=[], # initially empty
        api=None,
        refresh_models_fn=mock_refresh
    )
    models_registry.set_provider(provider)
    
    await models_registry.refresh("dynamic")
    
    models_list = models_registry.get_models("dynamic")
    assert len(models_list) == 1
    assert models_list[0].id == "dynamic-1"
