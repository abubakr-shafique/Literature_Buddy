"""Document data models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    text: str
    chunk_id: int
    source: str
    metadata: Optional[dict] = None
    
    def __str__(self) -> str:
        return self.text
    
    def __len__(self) -> int:
        return len(self.text)