"""
Dynamic model registry.
Reads providers and models from models.json configuration file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .api.openai_completions import openai_completions_api
from .auth.helpers import env_api_key_auth
from .auth.types import ProviderAuth
from .models import Models, Provider, create_provider
from .types import Model, ModelCost

DEFAULT_MODELS_JSON = {
    "providers": [
        {
            "id": "deepseek",
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "api": "openai-completions",
            "env_key": "DEEPSEEK_API_KEY",
            "models": [
                {
                    "id": "deepseek-v4-pro",
                    "name": "DeepSeek V4 Pro",
                    "reasoning": True,
                    "input": ["text"],
                    "cost": {"input": 2.0, "output": 8.0, "cache_read": 0.2, "cache_write": 2.0},
                    "context_window": 128000,
                    "max_tokens": 8192
                },
                {
                    "id": "deepseek-chat",
                    "name": "DeepSeek Chat (V3)",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0.14, "output": 0.28, "cache_read": 0.014, "cache_write": 0.14},
                    "context_window": 64000,
                    "max_tokens": 8192
                }
            ]
        },
        {
            "id": "openai",
            "name": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "api": "openai-completions",
            "env_key": "OPENAI_API_KEY",
            "models": [
                {
                    "id": "gpt-4o",
                    "name": "GPT-4o",
                    "reasoning": False,
                    "input": ["text", "image"],
                    "cost": {"input": 2.50, "output": 10.00, "cache_read": 1.25, "cache_write": 2.50},
                    "context_window": 128000,
                    "max_tokens": 16384
                }
            ]
        },
        {
            "id": "ollama",
            "name": "Ollama (Local)",
            "base_url": "http://127.0.0.1:11434/v1",
            "api": "openai-completions",
            "env_key": "OLLAMA_API_KEY",
            "models": []
        }
    ]
}

def parse_model(provider_id: str, provider_api: str, provider_base_url: str, data: dict[str, Any]) -> Model:
    cost_data = data.get("cost", {})
    return Model(
        id=data["id"],
        name=data["name"],
        api=provider_api,
        provider=provider_id,
        base_url=provider_base_url,
        reasoning=data.get("reasoning", False),
        input=data.get("input", ["text"]),
        cost=ModelCost(
            input=cost_data.get("input", 0.0),
            output=cost_data.get("output", 0.0),
            cache_read=cost_data.get("cache_read", 0.0),
            cache_write=cost_data.get("cache_write", 0.0)
        ),
        context_window=data.get("context_window", 8192),
        max_tokens=data.get("max_tokens", 4096),
        headers=data.get("headers", {}),
        compat=data.get("compat", {})
    )

def parse_provider(data: dict[str, Any]) -> Provider:
    provider_id = data["id"]
    api_type = data["api"]
    
    # Map API string to implementation
    api_impl = None
    if api_type == "openai-completions":
        api_impl = openai_completions_api()
    else:
        raise ValueError(f"Unknown API type: {api_type}")
        
    env_key = data.get("env_key")
    auth = ProviderAuth()
    if env_key:
        auth.api_key = env_api_key_auth(f"{data['name']} API Key", [env_key])
        
    base_url = data.get("base_url")
    models = [parse_model(provider_id, api_type, base_url, m) for m in data.get("models", [])]
    
    # For Ollama (or similar), we might need to inject a refresh_models_fn
    refresh_fn = None
    if provider_id == "ollama":
        import httpx
        async def fetch_ollama_models() -> list[Model]:
            nonlocal base_url, provider_id, api_type
            try:
                # the base_url is typically http://.../v1, we need to strip /v1 to hit /api/tags
                host = base_url.replace("/v1", "")
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{host}/api/tags", timeout=5.0)
                    resp.raise_for_status()
                    tags = resp.json().get("models", [])
                
                fetched_models = []
                for tag in tags:
                    tag_name = tag["name"]
                    # For a tag like "qwen2.5:32b", make it "Qwen2.5:32b (Local)"
                    display_name = f"{tag_name.capitalize()} (Local)"
                    fetched_models.append(Model(
                        id=tag_name,
                        name=display_name,
                        api=api_type,
                        provider=provider_id,
                        base_url=base_url,
                        reasoning=False,
                        input=["text"],
                        cost=ModelCost(0, 0, 0, 0),
                        context_window=8192,
                        max_tokens=4096
                    ))
                return fetched_models
            except Exception:
                return []
        refresh_fn = fetch_ollama_models

    return create_provider(
        id=provider_id,
        name=data["name"],
        auth=auth,
        models=models,
        api=api_impl,
        base_url=base_url,
        refresh_models_fn=refresh_fn
    )

class ModelsRegistry:
    def __init__(self, config_path: str | Path = "models.json"):
        self.config_path = Path(config_path)

    def load_models(self) -> Models:
        if not self.config_path.exists():
            # Auto-generate default configuration
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_MODELS_JSON, f, indent=2)
                
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = DEFAULT_MODELS_JSON
            
        models_container = Models()
        for prov_data in data.get("providers", []):
            try:
                provider = parse_provider(prov_data)
                models_container.set_provider(provider)
            except Exception as e:
                print(f"Failed to load provider {prov_data.get('id')}: {e}")
                
        return models_container
