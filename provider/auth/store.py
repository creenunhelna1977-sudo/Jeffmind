"""
Credential store implementations.
"""
from __future__ import annotations

import asyncio
from typing import Dict

from .types import Credential, CredentialStore


class InMemoryCredentialStore(CredentialStore):
    """
    A simple in-memory credential store for development/testing.
    In a real app, you would replace this with a secure keychain or database.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Credential] = {}
        self._lock = asyncio.Lock()

    async def read(self, provider: str) -> Credential | None:
        async with self._lock:
            return self._store.get(provider)

    async def write(self, provider: str, credential: Credential) -> None:
        async with self._lock:
            self._store[provider] = credential

    async def delete(self, provider: str) -> None:
        async with self._lock:
            self._store.pop(provider, None)
