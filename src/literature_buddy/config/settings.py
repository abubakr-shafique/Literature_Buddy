"""Settings loader and configuration management."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class Settings:
    """Application settings container."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.app = config_dict.get("app", {})
        self.models = config_dict.get("models", {})
        self.rag = config_dict.get("rag", {})
        self.pdf = config_dict.get("pdf", {})
        self.ui = config_dict.get("ui", {})
    
    def get_model_path(self) -> Path:
        """Get the local model directory path."""
        path_str = self.models.get("local_model_path", "./models")
        return Path(path_str)
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration."""
        return self.models.get("embedding", {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.models.get("llm", {})
    
    def get_reranker_config(self) -> Dict[str, Any]:
        """Get reranker configuration."""
        return self.models.get("reranker", {})


def load_settings(config_path: Path) -> Settings:
    """Load settings from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return Settings(config)