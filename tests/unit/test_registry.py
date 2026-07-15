import os
import pytest
from pathlib import Path
from provider.registry import ModelsRegistry

@pytest.fixture
def temp_models_file(tmp_path):
    return tmp_path / "models.json"

def test_registry_generates_default_config(temp_models_file):
    registry = ModelsRegistry(temp_models_file)
    assert not temp_models_file.exists()
    
    models = registry.load_models()
    assert temp_models_file.exists()
    
    # DeepSeek and OpenAI should be loaded
    assert models.get_provider("deepseek") is not None
    assert models.get_provider("openai") is not None
    
    # Check a specific model
    deepseek_pro = models.get_model("deepseek", "deepseek-v4-pro")
    assert deepseek_pro is not None
    assert deepseek_pro.name == "DeepSeek V4 Pro"
    assert deepseek_pro.api == "openai-completions"

def test_registry_loads_custom_config(temp_models_file):
    import json
    custom_config = {
        "providers": [
            {
                "id": "custom",
                "name": "Custom Provider",
                "base_url": "https://custom.api",
                "api": "openai-completions",
                "models": [
                    {
                        "id": "custom-model",
                        "name": "Custom Model"
                    }
                ]
            }
        ]
    }
    with open(temp_models_file, "w", encoding="utf-8") as f:
        json.dump(custom_config, f)
        
    registry = ModelsRegistry(temp_models_file)
    models = registry.load_models()
    
    assert models.get_provider("custom") is not None
    custom_model = models.get_model("custom", "custom-model")
    assert custom_model is not None
    assert custom_model.name == "Custom Model"
    assert custom_model.provider == "custom"
