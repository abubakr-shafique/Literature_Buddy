"""Document loader for PDF files."""

from pathlib import Path
from typing import Optional, List
import os


class DocumentLoader:
    """Loads PDF documents."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        if not self.file_path.suffix.lower() == '.pdf':
            raise ValueError(f"Expected PDF file, got: {self.file_path.suffix}")
    
    def load(self) -> str:
        """Load the entire PDF as text."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(self.file_path)
            text = ""
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text
            
        except ImportError:
            raise ImportError(
                "PyMuPDF not installed. Run: pip install PyMuPDF"
            )
    
    def get_metadata(self) -> dict:
        """Get PDF metadata."""
        try:
            import fitz
            
            doc = fitz.open(self.file_path)
            metadata = {
                "path": str(self.file_path),
                "pages": len(doc),
                "metadata": doc.metadata,
            }
            doc.close()
            return metadata
            
        except ImportError:
            raise ImportError("PyMuPDF not installed")