"""
API base utilities, including lazy_api.
"""
from __future__ import annotations

import importlib
from typing import AsyncGenerator

from ..models import ProviderStreams
from ..types import Context, Model, SimpleStreamOptions, StreamEvent, StreamOptions


def lazy_api(module_name: str, factory_name: str) -> ProviderStreams:
    """
    Lazily loads an API implementation to avoid importing heavy SDKs 
    (like openai or anthropic) until they are actually needed.
    """
    _instance: ProviderStreams | None = None

    def _get_instance() -> ProviderStreams:
        nonlocal _instance
        if not _instance:
            module = importlib.import_module(module_name)
            factory = getattr(module, factory_name)
            _instance = factory()
        return _instance

    class LazyProviderStreams(ProviderStreams):
        async def stream(
            self, model: Model, context: Context, options: StreamOptions | None = None
        ) -> AsyncGenerator[StreamEvent, None]:
            instance = _get_instance()
            async for event in instance.stream(model, context, options):
                yield event

        async def stream_simple(
            self, model: Model, context: Context, options: SimpleStreamOptions | None = None
        ) -> AsyncGenerator[StreamEvent, None]:
            instance = _get_instance()
            async for event in instance.stream_simple(model, context, options):
                yield event

    return LazyProviderStreams()
