"""
Loss functions for imbalanced classification.

Design Philosophy:
- Google: Mathematically principled approach to class imbalance
- Amazon: Reduces misclassification of rare but critical categories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces the relative loss for well-classified examples, focusing
    training on hard negatives.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    Google: Mathematically principled approach to imbalance.
    Amazon: Reduces misclassification of rare but critical categories
            like Security Incidents.
    
    Example:
        >>> criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights tensor [num_classes]. If None, uniform weights.
            gamma: Focusing parameter. Higher = more focus on hard examples.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy without reduction
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    Helps prevent overconfident predictions and improves calibration.
    
    Google: Improves model calibration and generalization.
    Amazon: More reliable confidence scores for downstream decisions.
    
    Example:
        >>> criterion = LabelSmoothingLoss(num_classes=12, smoothing=0.1)
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None
    ):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Label smoothing factor (0.1 means 10% probability spread)
            weight: Optional class weights
        """
        super().__init__()
        
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"Smoothing must be in [0, 1), got {smoothing}")
        
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.weight = weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Label smoothing loss value
        """
        # Create smooth labels
        smooth_labels = torch.zeros_like(logits)
        smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute loss
        loss = -smooth_labels * log_probs
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            weight_per_sample = weight.gather(0, targets)
            loss = loss.sum(dim=-1) * weight_per_sample
        else:
            loss = loss.sum(dim=-1)
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combination of Focal Loss and Label Smoothing.
    
    Provides both focus on hard examples and calibration benefits.
    
    Example:
        >>> criterion = CombinedLoss(
        ...     num_classes=12,
        ...     alpha=class_weights,
        ...     gamma=2.0,
        ...     smoothing=0.1,
        ...     focal_weight=0.7
        ... )
    """
    
    def __init__(
        self,
        num_classes: int,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        focal_weight: float = 0.7
    ):
        """
        Initialize combined loss.
        
        Args:
            num_classes: Number of classes
            alpha: Class weights for focal loss
            gamma: Focal loss focusing parameter
            smoothing: Label smoothing factor
            focal_weight: Weight for focal loss (1 - this for smoothing)
        """
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.smoothing_loss = LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=smoothing,
            weight=alpha
        )
        self.focal_weight = focal_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss."""
        focal = self.focal_loss(logits, targets)
        smoothing = self.smoothing_loss(logits, targets)
        
        return self.focal_weight * focal + (1 - self.focal_weight) * smoothing
