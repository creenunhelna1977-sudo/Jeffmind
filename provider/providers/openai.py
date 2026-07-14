"""
OpenAI provider implementation.
"""
from __future__ import annotations

from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from ..types import Model, ModelCost
from ..api.openai_completions import openai_completions_api


OPENAI_MODELS = [
    Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(
            input=2.50,
            output=10.00,
            cache_read=1.25,
            cache_write=2.50,
        ),
        context_window=128000,
        max_tokens=4096,
    ),
]


def openai_provider() -> Provider:
    return create_provider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"])
        ),
        models=OPENAI_MODELS,
        api=openai_completions_api(),
    )
