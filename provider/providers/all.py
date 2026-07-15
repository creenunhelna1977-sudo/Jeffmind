"""
Registry for all built-in providers.
"""
from __future__ import annotations

from ..models import Models
from ..registry import ModelsRegistry

_registry = ModelsRegistry()

def builtin_models() -> Models:
    """
    Creates a Models instance pre-populated by loading models.json.
    """
    return _registry.load_models()
