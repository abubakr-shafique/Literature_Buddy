"""Vector store for document embeddings."""

from typing import List, Tuple
import numpy as np

from ..document.models import DocumentChunk
from ..models.backends import EmbeddingBackend


class VectorStore:
    """Simple in-memory vector store using FAISS."""
    
    def __init__(
        self,
        chunks: List[DocumentChunk],
        embedding_backend: EmbeddingBackend
    ):
        self.chunks = chunks
        self.embedding_backend = embedding_backend
        
        # Build index
        try:
            import faiss
            
            # Encode all chunks
            texts = [chunk.text for chunk in chunks]
            self.embeddings = embedding_backend.encode(texts)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            
        except ImportError:
            raise ImportError(
                "faiss not installed. Run: pip install faiss-cpu"
            )
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Search for similar chunks."""
        # Encode query
        query_embedding = self.embedding_backend.encode_query(query)
        query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return as list of tuples
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((int(idx), float(dist)))
        
        return results