# src/literature_buddy/config/settings.py
"""Pydantic-based configuration for Literature Buddy.

Loads config/settings.yaml (path overridable via LITERATURE_BUDDY_CONFIG),
expanded with ~ and environment variables for data directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: Literal["ollama", "llamacpp", "transformers"] = "ollama"
    model: str = "qwen3:14b"
    model_path: Optional[str] = None  # used by llamacpp/transformers
    quantization: str = "4bit"
    max_tokens: int = 2048
    temperature: float = 0.1
    context_window: int = 32768
    n_gpu_layers: int = -1  # llamacpp: -1 = all layers on GPU


class VLMConfig(BaseModel):
    provider: Literal["ollama", "transformers"] = "ollama"
    model: str = "qwen3-vl:8b"
    model_path: Optional[str] = None
    max_tokens: int = 1536


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    batch_size: int = 32
    device: str = "auto"


class RerankerConfig(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 6


class RetrievalConfig(BaseModel):
    top_k: int = 12
    bm25_weight: float = 0.3
    figure_regex_boost: bool = True


class DocumentConfig(BaseModel):
    extract_figures: bool = True
    extract_tables: bool = True
    min_figure_area_px: int = 40_000
    ocr_fallback: bool = False


class StorageConfig(BaseModel):
    data_dir: str = "~/.literature_buddy"
    papers_dir: str = "~/.literature_buddy/papers"
    figures_dir: str = "~/.literature_buddy/figures"
    index_dir: str = "~/.literature_buddy/indexes"

    def resolved(self, field: str) -> Path:
        path = Path(os.path.expandvars(getattr(self, field))).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path


class MemoryConfig(BaseModel):
    max_history_turns: int = 6
    summary_threshold: int = 4


class UIConfig(BaseModel):
    window_title: str = "Literature Buddy"
    pdf_default_zoom: float = 1.2


class AppSettings(BaseModel):
    """Root application settings."""

    profile: str = "16gb"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


def load_settings(config_path: Optional[str] = None) -> AppSettings:
    """Load settings from YAML, falling back to defaults.

    Search order: explicit path -> $LITERATURE_BUDDY_CONFIG ->
    ./config/settings.yaml -> built-in defaults.
    """
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    if env_path := os.environ.get("LITERATURE_BUDDY_CONFIG"):
        candidates.append(Path(env_path))
    candidates.append(Path("config/settings.yaml"))

    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return AppSettings.model_validate(data)
    return AppSettings()
