"""
Credential store implementations.
"""
from __future__ import annotations

import asyncio
from typing import Dict

from .types import Credential, CredentialStore
from .file_store import FileCredentialStore


class InMemoryCredentialStore(CredentialStore):
    """
    A simple in-memory credential store for development/testing.
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


# Default global store instance
_default_store: CredentialStore | None = None

def get_credential_store() -> CredentialStore:
    """
    Returns the default credential store (FileCredentialStore pointing to .credentials.json).
    """
    global _default_store
    if _default_store is None:
        _default_store = FileCredentialStore(".credentials.json")
    return _default_store
