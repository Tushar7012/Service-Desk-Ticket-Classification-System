"""
Unit tests for model components.
"""

import pytest
import torch
import numpy as np

from src.models import TicketClassifier, FocalLoss, LabelSmoothingLoss


class TestTicketClassifier:
    """Tests for TicketClassifier model."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return TicketClassifier(
            num_classes=5,
            model_name="distilbert-base-uncased",
            dropout=0.1
        )
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.num_classes == 5
        assert model.hidden_size == 768  # DistilBERT hidden size
    
    def test_forward_pass(self, model):
        """Test forward pass produces correct output shape."""
        batch_size = 4
        seq_length = 128
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 5)
    
    def test_predict_proba(self, model):
        """Test probability prediction."""
        batch_size = 2
        seq_length = 64
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        proba = model.predict_proba(input_ids, attention_mask)
        
        assert proba.shape == (batch_size, 5)
        # Probabilities should sum to 1
        assert torch.allclose(proba.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        # All values should be between 0 and 1
        assert (proba >= 0).all() and (proba <= 1).all()
    
    def test_count_parameters(self, model):
        """Test parameter counting."""
        total_params = model.count_parameters(trainable_only=False)
        trainable_params = model.count_parameters(trainable_only=True)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_layer_freezing(self):
        """Test that layer freezing works."""
        model = TicketClassifier(
            num_classes=3,
            model_name="distilbert-base-uncased",
            freeze_bert_layers=2
        )
        
        # Check that some parameters are frozen
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        
        assert frozen_count > 0
        assert frozen_count < total_count


class TestFocalLoss:
    """Tests for Focal Loss."""
    
    def test_focal_loss_shape(self):
        """Test focal loss output shape."""
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative
    
    def test_focal_loss_with_weights(self):
        """Test focal loss with class weights."""
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_fn = FocalLoss(alpha=weights, gamma=2.0)
        
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0
    
    def test_focal_loss_focuses_on_hard_examples(self):
        """Test that focal loss down-weights easy examples."""
        loss_fn = FocalLoss(gamma=2.0)
        loss_fn_gamma0 = FocalLoss(gamma=0.0)  # Standard CE
        
        # Create easy example (high confidence correct)
        logits_easy = torch.tensor([[10.0, -10.0, -10.0]])
        targets_easy = torch.tensor([0])
        
        # Focal loss should be lower than standard CE for easy examples
        focal_easy = loss_fn(logits_easy, targets_easy)
        ce_easy = loss_fn_gamma0(logits_easy, targets_easy)
        
        assert focal_easy < ce_easy
    
    def test_focal_loss_no_reduction(self):
        """Test focal loss without reduction."""
        loss_fn = FocalLoss(gamma=2.0, reduction='none')
        
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.shape == (8,)


class TestLabelSmoothingLoss:
    """Tests for Label Smoothing Loss."""
    
    def test_label_smoothing_shape(self):
        """Test label smoothing loss output shape."""
        loss_fn = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
        
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_smoothing_zero_equals_ce(self):
        """Test that zero smoothing equals cross entropy."""
        smoothing_loss = LabelSmoothingLoss(num_classes=5, smoothing=0.0)
        
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        smooth = smoothing_loss(logits, targets)
        ce = torch.nn.functional.cross_entropy(logits, targets)
        
        assert torch.allclose(smooth, ce, atol=1e-5)
    
    def test_invalid_smoothing_raises(self):
        """Test that invalid smoothing values raise errors."""
        with pytest.raises(ValueError):
            LabelSmoothingLoss(num_classes=5, smoothing=1.0)
        
        with pytest.raises(ValueError):
            LabelSmoothingLoss(num_classes=5, smoothing=-0.1)
