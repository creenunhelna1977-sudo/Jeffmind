"""
DeepSeek provider implementation.
"""
from __future__ import annotations

from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from ..types import Model, ModelCost
from ..api.openai_completions import openai_completions_api


DEEPSEEK_MODELS = [
    Model(
        id="deepseek-v4-pro",           # 假设未来它叫这个名字
        name="DeepSeek V4 Pro",
        api="openai-completions",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        reasoning=True,                 # 如果它是个思考链模型就设为 True
        input=["text"],                 # 假设 V4 支持发图片了，这里就能配上 "image"
        cost=ModelCost(                 # 去官网查一下它的计费标准填进来
            input=2.0, output=8.0, cache_read=0.2, cache_write=2.0
        ),
        context_window=128000,
        max_tokens=8192,
    ),
    Model(
        id="deepseek-v4-flash",           # 假设未来它叫这个名字
        name="DeepSeek V4 Flash",
        api="openai-completions",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        reasoning=True,                 # 如果它是个思考链模型就设为 True
        input=["text"],                 # 假设 V4 支持发图片了，这里就能配上 "image"
        cost=ModelCost(                 # 去官网查一下它的计费标准填进来
            input=2.0, 
            output=8.0, 
            cache_read=0.2, 
            cache_write=2.0
        ),
        context_window=128000,
        max_tokens=8192,
    ),
    Model(
        id="deepseek-chat",
        name="DeepSeek Chat (V3)",
        api="openai-completions",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(
            input=0.14,
            output=0.28,
            cache_read=0.014,
            cache_write=0.14,
        ),
        context_window=64000,
        max_tokens=8192,
    ),
    Model(
        id="deepseek-reasoner",
        name="DeepSeek Reasoner (R1)",
        api="openai-completions",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(
            input=0.55,
            output=2.19,
            cache_read=0.14,
            cache_write=0.55,
        ),
        context_window=64000,
        max_tokens=8192,
    ),
]


def deepseek_provider() -> Provider:
    return create_provider(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"])
        ),
        models=DEEPSEEK_MODELS,
        api=openai_completions_api(),
    )
