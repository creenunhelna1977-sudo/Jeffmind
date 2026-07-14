"""
Models, Provider interfaces and registry.
Mirrors pi/packages/ai/src/models.ts
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Awaitable, Callable, Protocol, runtime_checkable

from .auth.resolve import resolve_provider_auth
from .auth.store import InMemoryCredentialStore
from .auth.types import AuthResult, CredentialStore, ProviderAuth, ModelAuth
from .types import (
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StreamEvent,
    StreamOptions,
    Usage,
)


class ModelsError(Exception):
    def __init__(self, operation: str, message: str):
        super().__init__(f"Models {operation} error: {message}")
        self.operation = operation
        self.message = message


@runtime_checkable
class ProviderStreams(Protocol):
    """Protocol for API implementations (e.g. openai_completions)."""
    
    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        ...

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        ...


class Provider(Protocol):
    """Protocol for a provider instance (e.g., OpenAI, Anthropic)."""
    id: str
    name: str
    base_url: str | None
    auth: ProviderAuth

    def get_models(self) -> list[Model]:
        ...

    async def refresh_models(self) -> None:
        ...

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        ...

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        ...


def create_provider(
    id: str,
    name: str,
    auth: ProviderAuth,
    models: list[Model],
    api: ProviderStreams | dict[str, ProviderStreams],
    base_url: str | None = None,
    refresh_models_fn: Callable[[], Awaitable[list[Model]]] | None = None,
) -> Provider:
    """
    Factory to create a Provider instance.
    """
    _models = list(models)
    _inflight_refresh: asyncio.Task | None = None

    def _get_api(model: Model) -> ProviderStreams | None:
        if isinstance(api, dict):
            return api.get(model.api)
        return api

    class ProviderImpl:
        def __init__(self):
            self.id = id
            self.name = name
            self.base_url = base_url
            self.auth = auth

        def get_models(self) -> list[Model]:
            return list(_models)

        async def refresh_models(self) -> None:
            nonlocal _models, _inflight_refresh
            if not refresh_models_fn:
                return
                
            if _inflight_refresh and not _inflight_refresh.done():
                await _inflight_refresh
                return

            async def _do_refresh():
                nonlocal _models
                try:
                    new_models = await refresh_models_fn()
                    _models = list(new_models)
                except Exception as e:
                    # Log error in a real app
                    pass

            _inflight_refresh = asyncio.create_task(_do_refresh())
            await _inflight_refresh

        async def stream(
            self, model: Model, context: Context, options: StreamOptions | None = None
        ) -> AsyncGenerator[StreamEvent, None]:
            streams = _get_api(model)
            if not streams:
                raise ModelsError("stream", f"Provider {id} has no API implementation for '{model.api}'")
            
            async for event in streams.stream(model, context, options):
                yield event

        async def stream_simple(
            self, model: Model, context: Context, options: SimpleStreamOptions | None = None
        ) -> AsyncGenerator[StreamEvent, None]:
            streams = _get_api(model)
            if not streams:
                raise ModelsError("stream_simple", f"Provider {id} has no API implementation for '{model.api}'")
                
            async for event in streams.stream_simple(model, context, options):
                yield event

    return ProviderImpl()


class Models:
    """
    The main entry point for the provider system.
    Manages providers, credentials, and routing.
    """
    def __init__(self):
        self._providers: dict[str, Provider] = {}
        self._credentials: CredentialStore = InMemoryCredentialStore()

    @property
    def credentials(self) -> CredentialStore:
        return self._credentials

    def set_provider(self, provider: Provider) -> None:
        self._providers[provider.id] = provider

    def delete_provider(self, id: str) -> None:
        self._providers.pop(id, None)

    def clear_providers(self) -> None:
        self._providers.clear()

    def get_providers(self) -> list[Provider]:
        return list(self._providers.values())

    def get_provider(self, id: str) -> Provider | None:
        return self._providers.get(id)

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        if provider_id:
            provider = self._providers.get(provider_id)
            return provider.get_models() if provider else []
            
        all_models = []
        for provider in self._providers.values():
            all_models.extend(provider.get_models())
        return all_models

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        provider = self._providers.get(provider_id)
        if not provider:
            return None
        for model in provider.get_models():
            if model.id == model_id:
                return model
        return None

    async def refresh(self, provider_id: str | None = None) -> None:
        if provider_id:
            provider = self._providers.get(provider_id)
            if provider:
                await provider.refresh_models()
        else:
            tasks = [p.refresh_models() for p in self._providers.values()]
            if tasks:
                await asyncio.gather(*tasks)

    async def get_auth(self, model: Model, options: StreamOptions | None = None) -> AuthResult | None:
        provider = self._providers.get(model.provider)
        if not provider:
            return None
        return await resolve_provider_auth(provider, model, Context(messages=[]), options, self._credentials)

    def _merge_auth_options(self, options: StreamOptions | None, auth_result: AuthResult | None) -> StreamOptions:
        import dataclasses
        if not options:
            options = StreamOptions()
        else:
            # Create a shallow copy to not mutate the caller's options
            options = dataclasses.replace(options)
            
        if auth_result and auth_result.auth:
            if auth_result.auth.api_key:
                options.api_key = auth_result.auth.api_key
            if auth_result.auth.headers:
                if not options.headers:
                    options.headers = {}
                options.headers.update(auth_result.auth.headers)
                
        return options

    async def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        provider = self._providers.get(model.provider)
        if not provider:
            raise ModelsError("stream", f"Provider not found: {model.provider}")
            
        auth_result = await resolve_provider_auth(provider, model, context, options, self._credentials)
        if not auth_result and not getattr(options, 'api_key', None):
            # We don't raise here immediately, some providers might work without auth
            # or the API implementation might throw a clearer error.
            pass
            
        merged_options = self._merge_auth_options(options, auth_result)
        
        async for event in provider.stream(model, context, merged_options):
            yield event

    async def complete(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessage:
        message = None
        async for event in self.stream(model, context, options):
            if event.type == "done":
                message = event.message
            elif event.type == "error":
                message = event.error
        
        if not message:
            raise ModelsError("complete", "Stream completed without yielding a done/error event")
        return message

    async def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        provider = self._providers.get(model.provider)
        if not provider:
            raise ModelsError("stream_simple", f"Provider not found: {model.provider}")
            
        auth_result = await resolve_provider_auth(provider, model, context, options, self._credentials)
        merged_options = self._merge_auth_options(options, auth_result)
        # Hack to cast to SimpleStreamOptions (we just set the reasoning attr)
        if options and getattr(options, "reasoning", None) is not None:
            setattr(merged_options, "reasoning", options.reasoning)
            
        async for event in provider.stream_simple(model, context, merged_options): # type: ignore
            yield event

    async def complete_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None
    ) -> AssistantMessage:
        message = None
        async for event in self.stream_simple(model, context, options):
            if event.type == "done":
                message = event.message
            elif event.type == "error":
                message = event.error
                
        if not message:
            raise ModelsError("complete_simple", "Stream completed without yielding a done/error event")
        return message
