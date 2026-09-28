"""Context builder for RAG."""

from typing import List
from ..document.models import DocumentChunk


class ContextBuilder:
    """Builds context from retrieved chunks."""
    
    def __init__(self, max_context_length: int = 2048):
        self.max_context_length = max_context_length
    
    def build(self, chunks: List[DocumentChunk]) -> str:
        """Build context string from chunks."""
        if not chunks:
            return ""
        
        # Combine chunks
        context_parts = []
        total_length = 0
        
        for chunk in chunks:
            chunk_text = chunk.text.strip()
            if total_length + len(chunk_text) <= self.max_context_length:
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            else:
                break
        
        return "\n\n".join(context_parts)