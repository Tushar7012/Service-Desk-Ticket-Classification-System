"""
Model architectures for ticket classification.

Provides DistilBERT-based classifier and BiLSTM+Attention alternative.

Design Philosophy:
- Google: State-of-the-art performance, proper regularization
- Amazon: Latency-conscious, supports model optimization
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TicketClassifier(nn.Module):
    """
    Production ticket classifier with DistilBERT backbone.
    
    Architecture Decisions:
    - DistilBERT: Best quality/latency trade-off
    - Dropout layers: Prevents overfitting on imbalanced data
    - Two-layer classification head: Learns task-specific representations
    
    Google: Proper initialization, regularization, multi-task ready
    Amazon: Layer freezing for cost-efficient training, exportable
    
    Example:
        >>> model = TicketClassifier(num_classes=12)
        >>> logits = model(input_ids, attention_mask)
    """
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.3,
        freeze_bert_layers: int = 0,
        hidden_dim: int = 256
    ):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of target classes
            model_name: HuggingFace model name
            dropout: Dropout probability
            freeze_bert_layers: Number of transformer layers to freeze
            hidden_dim: Hidden dimension of classification head
        """
        super().__init__()
        
        from transformers import DistilBertModel
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained backbone
        logger.info(f"Loading backbone: {model_name}")
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Optionally freeze early layers (Amazon: reduces training cost)
        if freeze_bert_layers > 0:
            self._freeze_layers(freeze_bert_layers)
        
        # Classification head with regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),  # Smoother than ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights (Google: proper initialization matters)
        self._init_weights()
        
        logger.info(f"Model initialized. Params: {self.count_parameters():,}")
    
    def _freeze_layers(self, n_layers: int):
        """Freeze first n transformer layers."""
        for layer in self.bert.transformer.layer[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        logger.info(f"Froze first {n_layers} transformer layers")
    
    def _init_weights(self):
        """Xavier initialization for classification layers."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass returning logits.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Return probability distribution for confidence monitoring.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Probabilities [batch_size, num_classes]
        """
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get CLS embeddings for analysis/visualization.
        
        Useful for error analysis and understanding model behavior.
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return outputs.last_hidden_state[:, 0, :]


class BiLSTMAttentionClassifier(nn.Module):
    """
    Lightweight alternative when transformer latency is prohibitive.
    
    Amazon: For edge deployment or cost-sensitive environments.
    Google: Useful for ablation to demonstrate transformer benefits.
    
    Architecture:
    - Embedding layer (with optional pretrained weights)
    - Bidirectional LSTM with dropout
    - Self-attention pooling
    - Classification head
    
    Example:
        >>> model = BiLSTMAttentionClassifier(vocab_size=30000, num_classes=12)
        >>> logits = model(input_ids, lengths)
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize BiLSTM classifier.
        
        Args:
            vocab_size: Vocabulary size
            num_classes: Number of target classes
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embedding weights
            freeze_embeddings: Whether to freeze embedding layer
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Self-attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            lengths: Sequence lengths [batch_size]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Embed tokens
        embedded = self.embedding(input_ids)
        
        # Pack for variable length sequences (efficiency)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out,
            batch_first=True
        )
        
        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out).squeeze(-1)
        
        # Mask padding positions
        max_len = input_ids.size(1)
        mask = torch.arange(max_len, device=input_ids.device).expand(
            len(lengths), max_len
        ) < lengths.unsqueeze(1)
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        return self.classifier(context)
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for interpretability.
        
        Returns:
            (logits, attention_weights) tuple
        """
        embedded = self.embedding(input_ids)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        attn_weights = self.attention(lstm_out).squeeze(-1)
        
        max_len = input_ids.size(1)
        mask = torch.arange(max_len, device=input_ids.device).expand(
            len(lengths), max_len
        ) < lengths.unsqueeze(1)
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        logits = self.classifier(context)
        
        return logits, attn_weights
