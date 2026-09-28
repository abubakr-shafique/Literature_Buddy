"""RAG pipeline for question answering."""

from typing import List, Optional
from pathlib import Path

from ..document.parser import DocumentParser
from ..document.models import DocumentChunk
from ..models import ModelLoader
from .retriever import Retriever
from .context_builder import ContextBuilder


class RAGPipeline:
    """End-to-end RAG pipeline."""
    
    def __init__(
        self,
        model_loader: ModelLoader,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        self.model_loader = model_loader
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        self.chunks: List[DocumentChunk] = []
        self.retriever: Optional[Retriever] = None
        self.context_builder = ContextBuilder()
    
    def index_document(self, file_path: str) -> None:
        """Index a document for retrieval."""
        # Parse document
        parser = DocumentParser(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.chunks = parser.parse(file_path)
        
        # Create retriever with embeddings
        embedding_backend = self.model_loader.load_embedding_model()
        self.retriever = Retriever(
            chunks=self.chunks,
            embedding_backend=embedding_backend
        )
    
    def query(self, question: str) -> str:
        """Query the RAG pipeline."""
        if self.retriever is None:
            return "No document indexed. Please load a PDF first."
        
        # Retrieve relevant chunks
        relevant_chunks = self.retriever.retrieve(question, top_k=self.top_k)
        
        # Build context
        context = self.context_builder.build(relevant_chunks)
        
        # Generate answer
        llm = self.model_loader.load_llm()
        answer = llm.generate_response(
            query=question,
            context=context
        )
        
        return answer