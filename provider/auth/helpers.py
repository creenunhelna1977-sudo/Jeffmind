"""
Auth helper functions.
Mirrors pi/packages/ai/src/auth/helpers.ts
"""
from __future__ import annotations

import os
from typing import Awaitable, Callable

from ..types import Context, Model
from .types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthResult,
    ModelAuth,
    OAuthAuth,
    OAuthProvider,
)


def env_api_key_auth(name: str, env_vars: list[str]) -> ApiKeyAuth:
    """
    Creates an ApiKeyAuth that checks stored credentials first,
    then falls back to environment variables.
    """
    
    class EnvApiKeyAuthImpl:
        def __init__(self, name: str, env_vars: list[str]):
            self.name = name
            self.env_vars = env_vars

        async def resolve(
            self,
            model: Model,
            context: Context,
            credential: ApiKeyCredential | None = None
        ) -> AuthResult | None:
            # 1. Stored credential
            if credential and credential.key:
                return AuthResult(
                    auth=ModelAuth(api_key=credential.key),
                    source="stored credential"
                )

            # 2. Environment variables (via context.env if we had one, or os.environ for now)
            for var in self.env_vars:
                # In Pi, context.env(var) is used. For this basic port, we use os.getenv if context doesn't have it.
                # Assuming context.env is not defined yet, we'll just check os.environ.
                val = None
                # If we implement env access in Context:
                # if hasattr(context, "env") and callable(context.env):
                #     val = await context.env(var)
                
                if not val:
                    val = os.getenv(var)
                    
                if val:
                    return AuthResult(
                        auth=ModelAuth(api_key=val),
                        source=var
                    )

            return None

    return EnvApiKeyAuthImpl(name, env_vars)


def lazy_oauth(name: str, load_fn: Callable[[], Awaitable[OAuthProvider]]) -> OAuthAuth:
    """
    Creates an OAuthAuth that lazily loads the OAuth provider implementation.
    """
    class LazyOAuthAuthImpl:
        def __init__(self, name: str, load_fn: Callable[[], Awaitable[OAuthProvider]]):
            self.name = name
            self.load_fn = load_fn

        async def load(self) -> OAuthProvider:
            return await self.load_fn()

    return LazyOAuthAuthImpl(name, load_fn)
