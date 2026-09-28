"""Citation tracking for RAG responses."""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Citation:
    """Represents a citation."""
    chunk_id: int
    source: str
    text: str


class CitationTracker:
    """Tracks citations in RAG responses."""
    
    def __init__(self):
        self.citations: Dict[int, Citation] = {}
    
    def add_citation(self, chunk_id: int, source: str, text: str) -> None:
        """Add a citation."""
        self.citations[chunk_id] = Citation(
            chunk_id=chunk_id,
            source=source,
            text=text
        )
    
    def get_citation(self, chunk_id: int) -> Citation:
        """Get a citation by ID."""
        return self.citations.get(chunk_id)
    
    def format_citations(self) -> str:
        """Format all citations for display."""
        if not self.citations:
            return ""
        
        lines = ["\n\n**Sources:**"]
        for chunk_id, citation in sorted(self.citations.items()):
            lines.append(f"[{chunk_id + 1}] {citation.source}")
        
        return "\n".join(lines)