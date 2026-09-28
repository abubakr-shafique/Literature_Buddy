"""Document parser with chunking."""

from typing import List
from .loader import DocumentLoader
from .chunker import TextChunker
from .models import DocumentChunk


class DocumentParser:
    """Parses documents into chunks."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = TextChunker(chunk_size, chunk_overlap)
    
    def parse(self, file_path: str) -> List[DocumentChunk]:
        """Parse a document into chunks."""
        # Load document
        loader = DocumentLoader(file_path)
        text = loader.load()
        
        # Chunk the text
        chunks = self.chunker.chunk(text)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            document_chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    chunk_id=i,
                    source=str(file_path)
                )
            )
        
        return document_chunks