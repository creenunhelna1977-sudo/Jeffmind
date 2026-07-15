"""
File-based credential store.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from .types import Credential, CredentialStore, ApiKeyCredential, OAuthCredential


class FileCredentialStore(CredentialStore):
    """
    A credential store that reads and writes to a local JSON file.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._lock = asyncio.Lock()

    def _load(self) -> Dict[str, Any]:
        if not self.file_path.exists():
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _deserialize(self, data: dict[str, Any]) -> Credential | None:
        ctype = data.get("type")
        if ctype == "api_key":
            return ApiKeyCredential(key=data.get("key", ""))
        elif ctype == "oauth":
            return OAuthCredential(
                access_token=data.get("access_token", ""),
                refresh_token=data.get("refresh_token"),
                expires=data.get("expires", 0),
                metadata=data.get("metadata")
            )
        return None

    def _serialize(self, cred: Credential) -> dict[str, Any]:
        if isinstance(cred, ApiKeyCredential):
            return {"type": "api_key", "key": cred.key}
        elif isinstance(cred, OAuthCredential):
            return {
                "type": "oauth",
                "access_token": cred.access_token,
                "refresh_token": cred.refresh_token,
                "expires": cred.expires,
                "metadata": cred.metadata
            }
        return {}

    async def read(self, provider: str) -> Credential | None:
        async with self._lock:
            data = self._load()
            if provider in data:
                return self._deserialize(data[provider])
            return None

    async def write(self, provider: str, credential: Credential) -> None:
        async with self._lock:
            data = self._load()
            data[provider] = self._serialize(credential)
            self._save(data)

    async def delete(self, provider: str) -> None:
        async with self._lock:
            data = self._load()
            if provider in data:
                del data[provider]
                self._save(data)
