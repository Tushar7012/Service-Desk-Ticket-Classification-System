"""Feature representation and embedding modules."""

from src.features.embedding_factory import (
    EmbeddingStrategy,
    TFIDFEmbedding,
    TransformerEmbedding,
    EmbeddingFactory
)

__all__ = [
    "EmbeddingStrategy",
    "TFIDFEmbedding", 
    "TransformerEmbedding",
    "EmbeddingFactory"
]
