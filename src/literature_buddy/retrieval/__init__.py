"""Retrieval module (legacy, kept for compatibility)."""

from .retriever import Retriever
from .embeddings import EmbeddingModel
from .vector_store import VectorStore

__all__ = ["Retriever", "EmbeddingModel", "VectorStore"]