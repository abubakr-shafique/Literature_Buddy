"""ML model backends for embeddings, LLM, and reranking."""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np


class EmbeddingBackend:
    """Sentence embedding backend using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "./models/embeddings",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        
        # Lazy load
        self._model = None
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=self.device
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model
    
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return np.array(embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""
        return self.encode([query])[0]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._model is None:
            # Load model to get dimension
            _ = self.model
        return self._model.get_sentence_embedding_dimension()


class LLMBackend:
    """LLM backend using transformers for local inference."""
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cache_dir: str = "./models/llm",
        device: str = "cpu",
        max_context_length: int = 2048,
        max_new_tokens: int = 512
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        
        # Lazy load
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    @property
    def model(self):
        """Lazy load the LLM model."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                
                # Set pad token if not exists
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                
                # Load model
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype="auto",
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Create pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    max_new_tokens=self.max_new_tokens,
                    return_full_text=False
                )
                
            except ImportError:
                raise ImportError(
                    "transformers not installed. "
                    "Run: pip install transformers torch"
                )
        return self._model
    
    def generate_response(
        self,
        query: str,
        context: str = "",
        temperature: float = 0.7
    ) -> str:
        """Generate a response given query and context."""
        if self._pipeline is None:
            _ = self.model  # Load model
        
        # Build prompt
        if context:
            prompt = f"""<|system|>
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

<|user|>
{query}

<|assistant|>
"""
        else:
            prompt = f"""<|system|>
You are a helpful assistant.

<|user|>
{query}

<|assistant|>
"""
        
        # Generate
        try:
            outputs = self._pipeline(
                prompt,
                temperature=temperature,
                do_sample=temperature > 0.0,
                top_p=0.95,
                pad_token_id=self._tokenizer.eos_token_id
            )
            
            if outputs and len(outputs) > 0:
                return outputs[0]['generated_text'].strip()
            else:
                return "I couldn't generate a response. Please try again."
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        """Generate text from a prompt."""
        if self._pipeline is None:
            _ = self.model
        
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_tokens or self.max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=0.95,
            pad_token_id=self._tokenizer.eos_token_id
        )
        
        if outputs and len(outputs) > 0:
            return outputs[0]['generated_text'].strip()
        return ""


class RerankerBackend:
    """Cross-encoder reranker for re-ranking retrieved documents."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_dir: str = "./models/reranker",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        
        # Lazy load
        self._model = None
    
    @property
    def model(self):
        """Lazy load the reranker model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device=self.device
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """
        Rerank documents based on relevance to query.
        
        Returns:
            List of (index, score, document) tuples, sorted by score descending.
        """
        if not documents:
            return []
        
        # Create pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Create indexed results
        results = [(i, float(score), doc) 
                   for i, (score, doc) in enumerate(zip(scores, documents))]
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return results[:top_k]