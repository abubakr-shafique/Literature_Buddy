"""Typed configuration.

Resolution order (later wins):
    built-in defaults  ->  profile (bundled YAML)  ->  user config file  ->  LB_* env vars
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from platformdirs import user_config_dir
from pydantic import BaseModel, ConfigDict, Field

PROFILE_DIR = Path(__file__).parent / "profiles"
DEFAULT_PROFILE = "16gb"

Provider = Literal["ollama", "transformers", "llama_cpp", "openai_compat"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelConfig(_Base):
    """Settings for one generative model (text LLM or vision-language model)."""

    enabled: bool = True
    provider: Provider = "ollama"
    model: str = ""  # Ollama tag, HF repo id, or served model name
    model_path: str | None = None  # local directory (transformers) or .gguf file (llama.cpp)
    gguf_file: str | None = None  # llama.cpp: file name inside a HF repo
    quantization: Literal["none", "8bit", "4bit"] = "4bit"  # transformers only
    base_url: str = "http://localhost:11434"  # ollama / openai_compat
    api_key: str | None = None
    context_length: int = 8192
    max_new_tokens: int = 1024
    temperature: float = 0.2
    n_gpu_layers: int = -1  # llama.cpp
    keep_alive: str = "10m"  # ollama
    think: bool | None = None  # None: do not send; False: disable reasoning mode where supported
    same_as_llm: bool = False  # VLM only: reuse the LLM backend (natively multimodal models)
    max_image_side: int = 1568  # VLM only: downscale figures above this size
    max_images: int = 2  # VLM only: figures analysed per question


class EmbeddingConfig(_Base):
    provider: Literal["sentence_transformers", "ollama", "hashing"] = "sentence_transformers"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    model_path: str | None = None
    base_url: str = "http://localhost:11434"
    device: str = "auto"  # auto | cuda | cpu
    batch_size: int = 16
    query_prompt: str = ""  # prefix applied to queries only (instruction-aware models)


class RetrievalConfig(_Base):
    top_k: int = 8
    candidates: int = 30
    hybrid: bool = True
    rrf_k: int = 60
    rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_path: str | None = None
    max_context_tokens: int = 5500
    max_visuals: int = 3


class DocumentConfig(_Base):
    ocr: bool = True  # only used for pages without a text layer (needs Tesseract)
    extract_figures: bool = True
    extract_tables: bool = True
    extract_equations: bool = True
    figure_dpi: int = 170
    max_chunk_words: int = 260
    min_chunk_words: int = 60
    max_download_mb: int = 100


class ConversationConfig(_Base):
    history_tokens: int = 1500
    rolling_summary: bool = True
    rewrite_queries: bool = True


class AppConfig(_Base):
    profile: str = DEFAULT_PROFILE
    llm: ModelConfig = Field(default_factory=ModelConfig)
    vlm: ModelConfig = Field(default_factory=lambda: ModelConfig(same_as_llm=True))
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    data_dir: str = "./data"
    models_dir: str = "./models"
    offline: bool = False
    log_level: str = "INFO"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir).expanduser().resolve()


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """LB_LLM__MODEL=x  ->  {"llm": {"model": "x"}};  LB_OFFLINE=true -> {"offline": True}."""
    reserved = {"LB_PROFILE", "LB_CONFIG", "LB_API_KEY"}
    out: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith("LB_") or key in reserved:
            continue
        path = key[3:].lower().split("__")
        try:
            value: Any = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return out


def user_config_path() -> Path:
    return Path(user_config_dir("literature-buddy")) / "config.yaml"


def load_config(
    profile: str | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> AppConfig:
    """Build the effective configuration. Raises ValueError on unknown profiles/keys."""
    if environ is None:
        load_dotenv()
        environ = dict(os.environ)
    explicit = config_path or environ.get("LB_CONFIG")
    user_file: Path | None = None
    if explicit:
        user_file = Path(explicit)
        if not user_file.exists():
            raise FileNotFoundError(f"Config file not found: {user_file}")
    else:
        user_file = next((p for p in (Path("config.yaml"), user_config_path()) if p.exists()), None)
    user_data = _read_yaml(user_file) if user_file else {}

    name = profile or environ.get("LB_PROFILE") or user_data.get("profile") or DEFAULT_PROFILE
    profile_file = PROFILE_DIR / f"{name}.yaml"
    if not profile_file.exists():
        raise ValueError(f"Unknown profile '{name}'. Available: {', '.join(available_profiles())}")

    data = _read_yaml(profile_file)
    user_data.pop("profile", None)
    data = _deep_merge(data, user_data)
    data["profile"] = name

    data = _deep_merge(data, _env_overrides(environ))
    if overrides:
        data = _deep_merge(data, overrides)

    cfg = AppConfig.model_validate(data)
    api_key = environ.get("LB_API_KEY")
    if api_key:
        cfg.llm.api_key = cfg.llm.api_key or api_key
        cfg.vlm.api_key = cfg.vlm.api_key or api_key
    return cfg


def save_user_config(patch: dict[str, Any], path: Path | None = None) -> Path:
    """Merge `patch` into the user config file (created if missing)."""
    path = path or user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _read_yaml(path) if path.exists() else {}
    path.write_text(yaml.safe_dump(_deep_merge(current, patch), sort_keys=False), encoding="utf-8")
    return path
