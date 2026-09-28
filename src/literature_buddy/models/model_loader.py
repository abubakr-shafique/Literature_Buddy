"""Model loader for local ML models."""

from pathlib import Path
from typing import Optional, Dict, Any
import os

from .backends import EmbeddingBackend, LLMBackend, RerankerBackend
from .config.settings import Settings


class ModelLoader:
    """Loads and manages local ML models."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_path = settings.get_model_path()
        
        # Model instances
        self.embedding_backend: Optional[EmbeddingBackend] = None
        self.llm_backend: Optional[LLMBackend] = None
        self.reranker_backend: Optional[RerankerBackend] = None
        
        # Ensure model directory exists
        self._ensure_model_directory()
    
    def _ensure_model_directory(self) -> None:
        """Create model directories if they don't exist."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.model_path / "embeddings").mkdir(exist_ok=True)
        (self.model_path / "llm").mkdir(exist_ok=True)
        (self.model_path / "reranker").mkdir(exist_ok=True)
    
    def load_embedding_model(self, force_download: bool = False) -> EmbeddingBackend:
        """Load embedding model from local directory."""
        if self.embedding_backend is not None and not force_download:
            return self.embedding_backend
        
        config = self.settings.get_embedding_config()
        model_name = config.get("model_name", "all-MiniLM-L6-v2")
        cache_dir = config.get("cache_dir", str(self.model_path / "embeddings"))
        device = config.get("device", "cpu")
        
        print(f"Loading embedding model: {model_name}")
        print(f"Cache directory: {cache_dir}")
        
        self.embedding_backend = EmbeddingBackend(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device
        )
        
        print("Embedding model loaded successfully")
        return self.embedding_backend
    
    def load_llm(self, force_download: bool = False) -> LLMBackend:
        """Load LLM from local directory."""
        if self.llm_backend is not None and not force_download:
            return self.llm_backend
        
        config = self.settings.get_llm_config()
        model_name = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        cache_dir = config.get("cache_dir", str(self.model_path / "llm"))
        device = config.get("device", "cpu")
        max_context = config.get("max_context_length", 2048)
        max_new_tokens = config.get("max_new_tokens", 512)
        
        print(f"Loading LLM: {model_name}")
        print(f"Cache directory: {cache_dir}")
        
        self.llm_backend = LLMBackend(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
            max_context_length=max_context,
            max_new_tokens=max_new_tokens
        )
        
        print("LLM loaded successfully")
        return self.llm_backend
    
    def load_reranker(self, force_download: bool = False) -> RerankerBackend:
        """Load reranker model from local directory."""
        if self.reranker_backend is not None and not force_download:
            return self.reranker_backend
        
        config = self.settings.get_reranker_config()
        model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        cache_dir = config.get("cache_dir", str(self.model_path / "reranker"))
        device = config.get("device", "cpu")
        
        print(f"Loading reranker: {model_name}")
        print(f"Cache directory: {cache_dir}")
        
        self.reranker_backend = RerankerBackend(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device
        )
        
        print("Reranker loaded successfully")
        return self.reranker_backend
    
    def load_all_models(self) -> None:
        """Load all models at once."""
        print("Loading all models...")
        self.load_embedding_model()
        self.load_llm()
        self.load_reranker()
        print("All models loaded successfully!")
    
    def is_model_downloaded(self, model_type: str) -> bool:
        """Check if a model is already downloaded."""
        model_dir = self.model_path / model_type
        return model_dir.exists() and len(list(model_dir.glob("*"))) > 0