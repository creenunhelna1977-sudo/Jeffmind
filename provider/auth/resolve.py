"""
Resolves authentication for a provider.
Mirrors pi/packages/ai/src/auth/resolve.ts
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from ..types import Context, Model, StreamOptions
from .types import AuthResult, CredentialStore, OAuthAuth, OAuthCredential

if TYPE_CHECKING:
    from ..models import Provider


# We use a per-provider lock to prevent concurrent OAuth token refreshes.
_provider_locks: dict[str, asyncio.Lock] = {}


async def resolve_stored_oauth(
    credentials: CredentialStore,
    provider_id: str,
    oauth_auth: OAuthAuth,
    stored_credential: OAuthCredential
) -> AuthResult | None:
    """
    Checks if an OAuth credential is valid, refreshing it if necessary using a double-checked lock.
    """
    now_ms = int(time.time() * 1000)
    
    # If not expired, use it directly.
    # Note: in a real implementation we might add a buffer (e.g., 5 minutes) to expiration.
    if now_ms < stored_credential.expires:
        oauth_provider = await oauth_auth.load()
        return AuthResult(
            auth=await oauth_provider.to_auth(stored_credential),
            source="OAuth"
        )

    # Need to refresh. Acquire the lock for this provider.
    lock = _provider_locks.setdefault(provider_id, asyncio.Lock())
    
    async with lock:
        # Double check: maybe another task already refreshed it while we waited for the lock.
        current_credential = await credentials.read(provider_id)
        if not current_credential or current_credential.type != "oauth":
            return None
            
        now_ms = int(time.time() * 1000)
        if now_ms < current_credential.expires:
            # Already refreshed!
            valid_credential = current_credential
        else:
            # Still expired, we must refresh it.
            oauth_provider = await oauth_auth.load()
            valid_credential = await oauth_provider.refresh(current_credential)
            await credentials.write(provider_id, valid_credential)
            
        oauth_provider = await oauth_auth.load()
        return AuthResult(
            auth=await oauth_provider.to_auth(valid_credential),
            source="OAuth"
        )


async def resolve_provider_auth(
    provider: Provider,
    model: Model,
    context: Context,
    options: StreamOptions | None,
    credentials: CredentialStore
) -> AuthResult | None:
    """
    Resolves the authentication to use for a model request.
    
    Resolution order:
    1. options.api_key (Explicit override for this request)
    2. Stored credential (OAuth or API Key)
    3. Ambient credentials (Environment variables)
    """
    
    # 1. Request-level override
    from .types import ModelAuth
    if options and options.api_key:
        return AuthResult(
            auth=ModelAuth(api_key=options.api_key),
            source="options.apiKey"
        )

    # 2. Check stored credentials
    stored = await credentials.read(provider.id)
    
    if stored:
        if stored.type == "oauth" and provider.auth.oauth:
            return await resolve_stored_oauth(credentials, provider.id, provider.auth.oauth, stored)
            
        if stored.type == "api_key" and provider.auth.api_key:
            return await provider.auth.api_key.resolve(model, context, stored)
            
    # 3. Ambient credentials (Environment)
    if provider.auth.api_key:
        return await provider.auth.api_key.resolve(model, context, None)
        
    return None
