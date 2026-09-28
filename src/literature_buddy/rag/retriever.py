"""Document retriever using vector similarity."""

from typing import List, Optional
import numpy as np

from ..document.models import DocumentChunk
from ..models.backends import EmbeddingBackend
from .vector_store import VectorStore


class Retriever:
    """Retrieves relevant document chunks."""
    
    def __init__(
        self,
        chunks: List[DocumentChunk],
        embedding_backend: EmbeddingBackend,
        top_k: int = 5
    ):
        self.chunks = chunks
        self.embedding_backend = embedding_backend
        self.top_k = top_k
        
        # Build vector store
        self.vector_store = VectorStore(
            chunks=chunks,
            embedding_backend=embedding_backend
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Retrieve relevant chunks for a query."""
        k = top_k or self.top_k
        
        # Get embeddings and similarities
        similarities = self.vector_store.search(query, top_k=k)
        
        # Return chunks sorted by similarity
        results = []
        for idx, score in similarities:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results