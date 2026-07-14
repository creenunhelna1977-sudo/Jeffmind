"""
Auth types for the provider layer.
Mirrors pi/packages/ai/src/auth/types.ts
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol, Union

from ..types import Model, Context


# ---------------------------------------------------------------------------
# Stored Credentials
# ---------------------------------------------------------------------------

@dataclass
class ApiKeyCredential:
    type: Literal["api_key"] = "api_key"
    key: str = ""


@dataclass
class OAuthCredential:
    type: Literal["oauth"] = "oauth"
    access_token: str = ""
    refresh_token: str | None = None
    expires: int = 0  # epoch ms
    metadata: dict[str, Any] | None = None


Credential = Union[ApiKeyCredential, OAuthCredential]


# ---------------------------------------------------------------------------
# Resolved Auth Result
# ---------------------------------------------------------------------------

@dataclass
class ModelAuth:
    api_key: str | None = None
    headers: dict[str, str] | None = None


@dataclass
class AuthResult:
    auth: ModelAuth
    source: str  # e.g., "OPENAI_API_KEY", "OAuth", "stored credential"


# ---------------------------------------------------------------------------
# Auth Implementations
# ---------------------------------------------------------------------------

class ApiKeyAuth(Protocol):
    name: str

    async def resolve(
        self,
        model: Model,
        context: Context,
        credential: ApiKeyCredential | None = None
    ) -> AuthResult | None:
        ...


class OAuthProvider(Protocol):
    name: str

    async def refresh(self, credential: OAuthCredential) -> OAuthCredential:
        ...

    async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
        ...


class OAuthAuth(Protocol):
    name: str

    async def load(self) -> OAuthProvider:
        ...


@dataclass
class ProviderAuth:
    api_key: ApiKeyAuth | None = None
    oauth: OAuthAuth | None = None


# ---------------------------------------------------------------------------
# Credential Store
# ---------------------------------------------------------------------------

class CredentialStore(Protocol):
    """Secure storage for API keys and OAuth tokens."""

    async def read(self, provider: str) -> Credential | None:
        ...

    async def write(self, provider: str, credential: Credential) -> None:
        ...

    async def delete(self, provider: str) -> None:
        ...
