"""
Data loading utilities for ticket classification.

Provides PyTorch Dataset and DataLoader creation.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple
import logging

from src.preprocessing import TicketPreprocessor

logger = logging.getLogger(__name__)


class TicketDataset(Dataset):
    """
    PyTorch Dataset for service desk tickets.
    
    Handles text preprocessing and tokenization.
    
    Example:
        >>> dataset = TicketDataset(
        ...     texts=["VPN not working", "Need password reset"],
        ...     labels=[3, 4],
        ...     tokenizer=tokenizer
        ... )
        >>> batch = dataset[0]
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        preprocessor: Optional[TicketPreprocessor] = None
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of ticket texts
            labels: List of category labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            preprocessor: Optional text preprocessor
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor or TicketPreprocessor()
        
        assert len(texts) == len(labels), "Texts and labels must have same length"
        logger.info(f"Created dataset with {len(texts)} samples")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Preprocess text
        if self.preprocessor:
            text = self.preprocessor.clean(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TicketDatasetFromDataFrame(Dataset):
    """
    Dataset that reads directly from a DataFrame.
    
    Supports subject + description combination.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        subject_col: str = 'subject',
        description_col: str = 'description',
        label_col: str = 'category',
        label_mapping: Optional[Dict[str, int]] = None,
        max_length: int = 256
    ):
        """
        Initialize dataset from DataFrame.
        
        Args:
            df: Pandas DataFrame with ticket data
            tokenizer: HuggingFace tokenizer
            subject_col: Column name for subject
            description_col: Column name for description
            label_col: Column name for labels
            label_mapping: Dict mapping label names to indices
            max_length: Maximum sequence length
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.subject_col = subject_col
        self.description_col = description_col
        self.label_col = label_col
        self.max_length = max_length
        self.preprocessor = TicketPreprocessor()
        
        # Create label mapping if not provided
        if label_mapping is None:
            unique_labels = df[label_col].unique()
            self.label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        else:
            self.label_mapping = label_mapping
        
        # Store inverse mapping for later use
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        logger.info(f"Dataset created: {len(df)} samples, {len(self.label_mapping)} classes")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self.df.iloc[idx]
        
        # Combine and preprocess fields
        subject = str(row.get(self.subject_col, ''))
        description = str(row.get(self.description_col, ''))
        text = self.preprocessor.combine_fields(subject, description)
        
        # Get label
        label_name = row[self.label_col]
        label = self.label_mapping[label_name]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def get_label_mapping(self) -> Dict[str, int]:
        """Return label to index mapping."""
        return self.label_mapping


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    num_workers: int = 0,
    subject_col: str = 'subject',
    description_col: str = 'description',
    label_col: str = 'category'
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train and validation data loaders.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        subject_col: Subject column name
        description_col: Description column name
        label_col: Label column name
        
    Returns:
        (train_loader, val_loader, label_mapping) tuple
    """
    # Create label mapping from training data
    unique_labels = train_df[label_col].unique()
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    # Create datasets
    train_dataset = TicketDatasetFromDataFrame(
        df=train_df,
        tokenizer=tokenizer,
        subject_col=subject_col,
        description_col=description_col,
        label_col=label_col,
        label_mapping=label_mapping,
        max_length=max_length
    )
    
    val_dataset = TicketDatasetFromDataFrame(
        df=val_df,
        tokenizer=tokenizer,
        subject_col=subject_col,
        description_col=description_col,
        label_col=label_col,
        label_mapping=label_mapping,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created loaders: train={len(train_loader)} batches, val={len(val_loader)} batches")
    
    return train_loader, val_loader, label_mapping
