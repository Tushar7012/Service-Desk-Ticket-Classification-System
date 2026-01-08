"""
Training script for ticket classifier.

Usage:
    python scripts/train.py --config configs/training_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from transformers import AutoTokenizer
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TicketClassifier
from src.training import create_data_loaders, TicketClassifierTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train ticket classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading datasets...")
    train_path = config['data']['train_path']
    val_path = config['data']['val_path']
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Load tokenizer
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create data loaders
    train_loader, val_loader, label_mapping = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_length']
    )
    
    # Create model
    logger.info("Initializing model...")
    model = TicketClassifier(
        num_classes=len(label_mapping),
        model_name=model_name,
        dropout=config['model']['dropout'],
        freeze_bert_layers=config['model'].get('freeze_bert_layers', 0)
    )
    
    # Create trainer config
    trainer_config = {
        **config['training'],
        'class_weights': config.get('class_weights'),
        'save_dir': config['checkpointing']['save_dir'],
        'project_name': config['logging']['project_name'],
        'experiment_name': config['logging']['experiment_name']
    }
    
    # Initialize trainer
    trainer = TicketClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
        use_wandb=not args.no_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    results = trainer.train()
    
    logger.info(f"Training complete! Best F1: {results['best_val_f1']:.4f}")
    logger.info(f"Total epochs: {results['total_epochs']}")


if __name__ == "__main__":
    main()
