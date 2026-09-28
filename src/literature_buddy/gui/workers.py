"""Worker threads for background tasks."""

from PySide6.QtCore import QThread, Signal, QObject
from typing import Optional


class ModelLoadingWorker(QObject):
    """Worker for loading ML models in background."""
    
    finished = Signal(object)  # Emits loaded model
    error = Signal(str)  # Emits error message
    progress = Signal(str)  # Emits progress message
    
    def __init__(self, model_loader, model_type: str):
        super().__init__()
        self.model_loader = model_loader
        self.model_type = model_type
    
    def run(self):
        """Load model in background."""
        try:
            self.progress.emit(f"Loading {self.model_type} model...")
            
            if self.model_type == "embedding":
                model = self.model_loader.load_embedding_model()
            elif self.model_type == "llm":
                model = self.model_loader.load_llm()
            elif self.model_type == "reranker":
                model = self.model_loader.load_reranker()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.finished.emit(model)
            
        except Exception as e:
            self.error.emit(str(e))


class DocumentIndexingWorker(QObject):
    """Worker for indexing documents in background."""
    
    finished = Signal(int)  # Emits number of chunks
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, rag_pipeline, file_path: str):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.file_path = file_path
    
    def run(self):
        """Index document in background."""
        try:
            self.progress.emit("Loading document...")
            self.rag_pipeline.index_document(self.file_path)
            self.finished.emit(len(self.rag_pipeline.chunks))
            
        except Exception as e:
            self.error.emit(str(e))


class QueryWorker(QObject):
    """Worker for RAG queries in background."""
    
    finished = Signal(str)  # Emits response
    error = Signal(str)
    
    def __init__(self, rag_pipeline, query: str):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.query = query
    
    def run(self):
        """Execute query in background."""
        try:
            response = self.rag_pipeline.query(self.query)
            self.finished.emit(response)
            
        except Exception as e:
            self.error.emit(str(e))