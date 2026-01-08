"""
Production training pipeline with experiment tracking.

Design Philosophy:
- Google: Full experiment reproducibility with wandb logging
- Amazon: Cost-efficient training with early stopping and checkpointing
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Callable
import logging
import os
from pathlib import Path
from tqdm import tqdm
import json

from src.models.losses import FocalLoss

logger = logging.getLogger(__name__)


class TicketClassifierTrainer:
    """
    Production training pipeline with experiment tracking.
    
    Google: Full experiment reproducibility with wandb logging.
    Amazon: Cost-efficient training with early stopping and checkpointing.
    
    Example:
        >>> trainer = TicketClassifierTrainer(model, train_loader, val_loader, config)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda",
        use_wandb: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict
            device: Device to train on
            use_wandb: Whether to use wandb for logging
        """
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Optimizer with weight decay (Google: regularization)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.get('max_epochs', 20)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.get('learning_rate', 2e-5),
            total_steps=total_steps,
            pct_start=config.get('warmup_ratio', 0.1)
        )
        
        # Loss function with class weights
        class_weights = config.get('class_weights')
        if class_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        self.criterion = FocalLoss(
            alpha=class_weights,
            gamma=config.get('focal_gamma', 2.0)
        )
        
        # Early stopping (Amazon: don't waste compute)
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.patience = config.get('patience', 5)
        
        # Gradient accumulation
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('save_dir', 'models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = []
        
        logger.info(f"Trainer initialized. Device: {device}, Steps: {total_steps}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.grad_accum_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1
        
        # Compute metrics
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, accuracy_score
        )
        
        metrics = {
            'val_loss': total_loss / max(num_batches, 1),
            'val_accuracy': accuracy_score(all_labels, all_preds),
            'val_f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'val_f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'val_precision_macro': precision_score(
                all_labels, all_preds, average='macro', zero_division=0
            ),
            'val_recall_macro': recall_score(
                all_labels, all_preds, average='macro', zero_division=0
            )
        }
        
        return metrics
    
    def train(self) -> Dict[str, any]:
        """
        Full training loop with wandb logging.
        
        Returns:
            Training results dictionary
        """
        # Initialize wandb if enabled
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.get('project_name', 'ticket-classifier'),
                    name=self.config.get('experiment_name', 'training-run'),
                    config=self.config
                )
            except ImportError:
                logger.warning("wandb not installed, disabling logging")
                self.use_wandb = False
        
        max_epochs = self.config.get('max_epochs', 20)
        logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            self.training_history.append(all_metrics)
            
            # Log to wandb
            if self.use_wandb:
                import wandb
                wandb.log(all_metrics)
            
            # Log to console
            logger.info(
                f"Epoch {epoch + 1}/{max_epochs} - "
                f"Loss: {train_metrics['train_loss']:.4f}, "
                f"Val F1: {val_metrics['val_f1_macro']:.4f}"
            )
            
            # Early stopping check (Amazon: cost efficiency)
            if val_metrics['val_f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['val_f1_macro']
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt", val_metrics)
                logger.info(f"New best model saved! F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", val_metrics)
        
        # Save final model
        self._save_checkpoint("final_model.pt", val_metrics)
        
        # Close wandb
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        return {
            'best_val_f1': self.best_val_f1,
            'total_epochs': self.current_epoch + 1,
            'history': self.training_history
        }
    
    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_f1': self.best_val_f1,
            'metrics': metrics
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    smoothing: float = 0.1
) -> List[float]:
    """
    Compute class weights using inverse frequency.
    
    Args:
        labels: List of label indices
        num_classes: Total number of classes
        smoothing: Smoothing factor to prevent extreme weights
        
    Returns:
        List of class weights
    """
    import numpy as np
    
    counts = np.bincount(labels, minlength=num_classes)
    frequencies = counts / counts.sum()
    
    # Inverse frequency with smoothing
    weights = 1.0 / (frequencies + smoothing)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return weights.tolist()
