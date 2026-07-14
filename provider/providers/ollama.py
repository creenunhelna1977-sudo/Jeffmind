"""
Ollama local provider implementation.
Ollama provides an OpenAI-compatible API endpoint by default.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..auth.types import ProviderAuth, ApiKeyAuth, AuthResult, ModelAuth, ApiKeyCredential
from ..models import Provider, create_provider
from ..types import Model, ModelCost
from ..api.openai_completions import openai_completions_api

if TYPE_CHECKING:
    from ..types import Context


# Ollama supports many models. We define a few common default ones here.
# In a full implementation, you could use `refresh_models_fn` to dynamically
# query `http://127.0.0.1:11434/api/tags` to populate this list.
OLLAMA_MODELS = [
    Model(
        id="llama3",
        name="Llama 3 (Local)",
        api="openai-completions",
        provider="ollama",
        base_url="http://127.0.0.1:11434/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0), # Local is free
        context_window=8192,
        max_tokens=4096,
    ),
    Model(
        id="qwen2.5:3b",
        name="Qwen 2.5 (Local)",
        api="openai-completions",
        provider="ollama",
        base_url="http://127.0.0.1:11434/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
        context_window=8192,
        max_tokens=4096,
    ),
    Model(
        id="deepseek-r1",
        name="DeepSeek R1 (Local)",
        api="openai-completions",
        provider="ollama",
        base_url="http://127.0.0.1:11434/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
        context_window=8192,
        max_tokens=4096,
        compat={"thinking_format": "deepseek", "requires_reasoning_content_on_assistant_messages": True}
    ),
]


class LocalDummyAuth(ApiKeyAuth):
    """Ollama doesn't require authentication, so we just provide a dummy key."""
    name = "Ollama Local Auth"

    async def resolve(
        self,
        model: Model,
        context: Context,
        credential: ApiKeyCredential | None = None
    ) -> AuthResult | None:
        return AuthResult(auth=ModelAuth(api_key="ollama"), source="local")


import httpx

async def _fetch_ollama_models() -> list[Model]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags", timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            
            models = []
            for m in data.get("models", []):
                model_name = m.get("name")
                # Simple heuristic to detect reasoning models
                is_reasoning = "deepseek-r1" in model_name.lower() or "-reasoner" in model_name.lower()
                
                models.append(Model(
                    id=model_name,
                    name=f"{model_name} (Local)",
                    api="openai-completions",
                    provider="ollama",
                    base_url="http://127.0.0.1:11434/v1",
                    reasoning=is_reasoning,
                    input=["text"],
                    cost=ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
                    context_window=8192,
                    max_tokens=4096,
                    compat={"thinking_format": "deepseek", "requires_reasoning_content_on_assistant_messages": True} if is_reasoning else {}
                ))
            return models
    except Exception:
        # Fallback to the hardcoded list if Ollama is offline or errors
        return OLLAMA_MODELS


def ollama_provider() -> Provider:
    return create_provider(
        id="ollama",
        name="Ollama",
        base_url="http://127.0.0.1:11434/v1",
        auth=ProviderAuth(
            api_key=LocalDummyAuth()
        ),
        models=OLLAMA_MODELS, # Initial sync fallback
        api=openai_completions_api(),
        refresh_models_fn=_fetch_ollama_models,
    )
