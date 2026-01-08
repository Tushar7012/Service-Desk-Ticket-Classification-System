"""
Feature representation strategies for ticket classification.

Provides multiple embedding approaches with a factory pattern for easy switching.

Design Philosophy:
- Google: Best representation quality, proper abstractions for experimentation
- Amazon: Latency and cost-conscious options, graceful degradation
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingStrategy(ABC):
    """
    Abstract base class for embedding strategies.
    
    Enables easy A/B testing between different representations.
    Google: Clean abstraction for experiment reproducibility.
    Amazon: Allows swapping embeddings based on latency requirements.
    """
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @property
    def name(self) -> str:
        """Return strategy name for logging."""
        return self.__class__.__name__


class TFIDFEmbedding(EmbeddingStrategy):
    """
    TF-IDF baseline embedding for fast prototyping and explainability.
    
    Google: Provides interpretable baseline for ablation studies.
    Amazon: Low latency, works on CPU, minimal infrastructure needs.
    
    Example:
        >>> tfidf = TFIDFEmbedding(max_features=5000)
        >>> tfidf.fit(train_texts)
        >>> embeddings = tfidf.encode(test_texts)
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: N-gram range (min, max)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            sublinear_tf: Apply log scaling to term frequency
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents='unicode',
            lowercase=True
        )
        self._fitted = False
        self._dim = max_features
    
    def fit(self, texts: List[str]) -> 'TFIDFEmbedding':
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts: Training texts
            
        Returns:
            self for chaining
        """
        self.vectorizer.fit(texts)
        self._fitted = True
        self._dim = len(self.vectorizer.vocabulary_)
        logger.info(f"TF-IDF fitted with vocabulary size: {self._dim}")
        return self
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to TF-IDF vectors."""
        if not self._fitted:
            raise RuntimeError("TF-IDF vectorizer must be fitted before encoding")
        return self.vectorizer.transform(texts).toarray()
    
    def get_embedding_dim(self) -> int:
        return self._dim
    
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        if not self._fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()


class TransformerEmbedding(EmbeddingStrategy):
    """
    Production transformer embeddings with optimization.
    
    Google: State-of-the-art representation quality.
    Amazon: Uses DistilBERT by default to balance quality vs. cost.
    
    Example:
        >>> embedder = TransformerEmbedding(model_name="distilbert-base-uncased")
        >>> embeddings = embedder.encode(texts)
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        batch_size: int = 32,
        device: str = "cuda",
        pooling_strategy: str = "mean"
    ):
        """
        Initialize transformer embedding extractor.
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
            device: Device to use ('cuda' or 'cpu')
            pooling_strategy: How to pool token embeddings ('mean', 'cls', 'max')
        """
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling_strategy = pooling_strategy
        
        # Determine device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device
        
        # Load model and tokenizer
        logger.info(f"Loading transformer model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self._dim = self.model.config.hidden_size
        logger.info(f"Transformer loaded. Embedding dim: {self._dim}, Device: {self.device}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to transformer embeddings.
        
        Uses batched inference for efficiency.
        """
        import torch
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Apply pooling strategy
                if self.pooling_strategy == "cls":
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooling_strategy == "max":
                    embeddings = outputs.last_hidden_state.max(dim=1)[0]
                else:  # mean pooling (default)
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    masked_embeddings = outputs.last_hidden_state * attention_mask
                    embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dim(self) -> int:
        return self._dim


class EmbeddingFactory:
    """
    Factory for creating embedding strategies.
    
    Simplifies switching between different embedding approaches.
    
    Example:
        >>> embedder = EmbeddingFactory.create("transformer", model_name="distilbert-base-uncased")
        >>> embeddings = embedder.encode(texts)
    """
    
    _strategies = {
        "tfidf": TFIDFEmbedding,
        "transformer": TransformerEmbedding,
    }
    
    @classmethod
    def create(cls, strategy_name: str, **kwargs) -> EmbeddingStrategy:
        """
        Create an embedding strategy by name.
        
        Args:
            strategy_name: Name of the strategy ('tfidf', 'transformer')
            **kwargs: Strategy-specific arguments
            
        Returns:
            Initialized embedding strategy
            
        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_name = strategy_name.lower()
        
        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy: {strategy_name}. Available: {available}"
            )
        
        return cls._strategies[strategy_name](**kwargs)
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a new embedding strategy."""
        cls._strategies[name.lower()] = strategy_class
