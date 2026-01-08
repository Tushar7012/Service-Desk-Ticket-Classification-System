"""
Error analysis framework for model debugging.

Design Philosophy:
- Google: Understanding failure modes for model improvement
- Amazon: Identifying errors that hurt customers most
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Qualitative error analysis for model debugging.
    
    Google: Understanding failure modes for model improvement.
    Amazon: Identifying errors that hurt customers most.
    
    Example:
        >>> analyzer = ErrorAnalyzer(class_names)
        >>> errors_df = analyzer.analyze_errors(texts, y_true, y_pred, y_proba)
    """
    
    def __init__(
        self,
        class_names: List[str],
        confidence_threshold: float = 0.7
    ):
        """
        Initialize error analyzer.
        
        Args:
            class_names: List of class names
            confidence_threshold: Threshold for low confidence
        """
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.idx_to_class = {i: name for i, name in enumerate(class_names)}
    
    def analyze_errors(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        sample_size: int = 100
    ) -> pd.DataFrame:
        """
        Generate detailed error analysis dataframe.
        
        Args:
            texts: Original text inputs
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            sample_size: Maximum errors to analyze
            
        Returns:
            DataFrame with error details
        """
        # Find error indices
        errors_mask = y_true != y_pred
        error_indices = np.where(errors_mask)[0]
        
        if len(error_indices) == 0:
            logger.info("No errors found!")
            return pd.DataFrame()
        
        # Prioritize high-confidence errors (most problematic)
        confidences = np.max(y_proba, axis=1)
        error_confidences = confidences[error_indices]
        
        # Sort by descending confidence (high-confidence errors first)
        sorted_indices = error_indices[np.argsort(-error_confidences)]
        
        if len(sorted_indices) > sample_size:
            sorted_indices = sorted_indices[:sample_size]
        
        # Build error records
        error_records = []
        
        for idx in sorted_indices:
            true_label = int(y_true[idx])
            pred_label = int(y_pred[idx])
            proba = y_proba[idx]
            
            record = {
                'index': int(idx),
                'text': texts[idx][:500],  # Truncate for readability
                'text_length': len(texts[idx]),
                'true_label': self.idx_to_class.get(true_label, str(true_label)),
                'predicted_label': self.idx_to_class.get(pred_label, str(pred_label)),
                'confidence': float(np.max(proba)),
                'true_class_prob': float(proba[true_label]),
                'prob_gap': float(np.max(proba) - proba[true_label]),
                'error_type': self._categorize_error(true_label, pred_label, proba),
                'second_best': self._get_second_best(proba, pred_label),
                'top_3_predictions': self._get_top_k(proba, 3)
            }
            
            error_records.append(record)
        
        df = pd.DataFrame(error_records)
        
        logger.info(f"Analyzed {len(df)} errors")
        return df
    
    def _categorize_error(
        self,
        true_label: int,
        pred_label: int,
        proba: np.ndarray
    ) -> str:
        """
        Categorize error type for systematic analysis.
        
        Error types:
        - High-confidence error: Model was very confident but wrong (needs more data/retraining)
        - Ambiguous prediction: True class had decent probability (might need multi-label)
        - Low-confidence error: Model wasn't sure (might benefit from human review)
        - Confusion pair: Common confusion between similar categories
        """
        confidence = float(np.max(proba))
        true_prob = float(proba[true_label])
        
        if confidence > 0.9:
            return "High-confidence error"
        elif true_prob > 0.3:
            return "Ambiguous prediction"
        elif confidence < 0.5:
            return "Low-confidence error"
        else:
            return "Standard misclassification"
    
    def _get_second_best(
        self,
        proba: np.ndarray,
        top_pred: int
    ) -> Dict:
        """Get second-best prediction."""
        sorted_indices = np.argsort(-proba)
        second_idx = sorted_indices[1] if sorted_indices[0] == top_pred else sorted_indices[0]
        
        return {
            'label': self.idx_to_class.get(int(second_idx), str(second_idx)),
            'probability': float(proba[second_idx])
        }
    
    def _get_top_k(self, proba: np.ndarray, k: int = 3) -> List[Dict]:
        """Get top-k predictions with probabilities."""
        sorted_indices = np.argsort(-proba)[:k]
        
        return [
            {
                'label': self.idx_to_class.get(int(idx), str(idx)),
                'probability': float(proba[idx])
            }
            for idx in sorted_indices
        ]
    
    def get_confusion_pairs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, str, int]]:
        """
        Find most common confusion pairs.
        
        Useful for identifying systematic model weaknesses.
        
        Returns:
            List of (true_class, pred_class, count) tuples
        """
        errors_mask = y_true != y_pred
        error_pairs = list(zip(y_true[errors_mask], y_pred[errors_mask]))
        
        # Count confusion pairs
        pair_counts = Counter(error_pairs)
        
        # Convert to readable format
        result = []
        for (true_idx, pred_idx), count in pair_counts.most_common(top_k):
            true_name = self.idx_to_class.get(int(true_idx), str(true_idx))
            pred_name = self.idx_to_class.get(int(pred_idx), str(pred_idx))
            result.append((true_name, pred_name, count))
        
        return result
    
    def get_error_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """
        Generate summary statistics about errors.
        
        Returns:
            Dictionary with error summary
        """
        errors_mask = y_true != y_pred
        error_indices = np.where(errors_mask)[0]
        
        if len(error_indices) == 0:
            return {"total_errors": 0}
        
        confidences = np.max(y_proba, axis=1)
        error_confidences = confidences[errors_mask]
        
        # Categorize errors
        error_types = []
        for idx in error_indices:
            error_type = self._categorize_error(
                int(y_true[idx]),
                int(y_pred[idx]),
                y_proba[idx]
            )
            error_types.append(error_type)
        
        type_counts = Counter(error_types)
        
        # Per-class error rates
        per_class_errors = {}
        for i, name in enumerate(self.class_names):
            class_mask = y_true == i
            if class_mask.sum() > 0:
                error_rate = float((y_pred[class_mask] != i).mean())
                per_class_errors[name] = {
                    'error_rate': error_rate,
                    'total': int(class_mask.sum()),
                    'errors': int((y_pred[class_mask] != i).sum())
                }
        
        return {
            'total_errors': int(len(error_indices)),
            'error_rate': float(len(error_indices) / len(y_true)),
            'error_confidence_stats': {
                'mean': float(error_confidences.mean()),
                'std': float(error_confidences.std()),
                'high_confidence_errors': int((error_confidences > 0.9).sum()),
                'medium_confidence_errors': int(
                    ((error_confidences > 0.5) & (error_confidences <= 0.9)).sum()
                ),
                'low_confidence_errors': int((error_confidences <= 0.5).sum())
            },
            'error_type_distribution': dict(type_counts),
            'per_class_error_rates': per_class_errors,
            'top_confusion_pairs': self.get_confusion_pairs(y_true, y_pred, top_k=5)
        }
    
    def generate_error_report(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> str:
        """Generate a human-readable error analysis report."""
        summary = self.get_error_summary(y_true, y_pred, y_proba)
        
        lines = [
            "=" * 60,
            "ERROR ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Total Errors: {summary['total_errors']} ({summary['error_rate']*100:.1f}%)",
            "",
            "Error Confidence Distribution:",
            f"  High (>90%): {summary['error_confidence_stats']['high_confidence_errors']}",
            f"  Medium (50-90%): {summary['error_confidence_stats']['medium_confidence_errors']}",
            f"  Low (<50%): {summary['error_confidence_stats']['low_confidence_errors']}",
            "",
            "Error Type Distribution:",
        ]
        
        for error_type, count in summary['error_type_distribution'].items():
            lines.append(f"  {error_type}: {count}")
        
        lines.extend([
            "",
            "Top Confusion Pairs:",
        ])
        
        for true_name, pred_name, count in summary['top_confusion_pairs']:
            lines.append(f"  {true_name} -> {pred_name}: {count}")
        
        lines.extend([
            "",
            "Per-Class Error Rates (sorted by error rate):",
        ])
        
        sorted_classes = sorted(
            summary['per_class_error_rates'].items(),
            key=lambda x: x[1]['error_rate'],
            reverse=True
        )
        
        for name, stats in sorted_classes:
            lines.append(
                f"  {name}: {stats['error_rate']*100:.1f}% "
                f"({stats['errors']}/{stats['total']})"
            )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
