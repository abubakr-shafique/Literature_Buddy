"""Text chunking utilities."""

from typing import List


class TextChunker:
    """Splits text into overlapping chunks."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if not text:
            return []
        
        # Simple character-based chunking
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            
            # Avoid infinite loop if overlap >= chunk_size
            if start >= len(text):
                break
        
        return chunks