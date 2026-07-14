"""
Registry for all built-in providers.
"""
from __future__ import annotations

from ..models import Models, Provider
from .deepseek import deepseek_provider
from .openai import openai_provider
from .ollama import ollama_provider


def builtin_providers() -> list[Provider]:
    """
    Returns a list of all built-in providers.
    """
    return [
        openai_provider(),
        deepseek_provider(),
        ollama_provider(),
    ]


def builtin_models() -> Models:
    """
    Creates a Models instance pre-populated with all built-in providers.
    """
    models = Models()
    for provider in builtin_providers():
        models.set_provider(provider)
    return models
