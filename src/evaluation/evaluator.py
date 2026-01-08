"""
Comprehensive model evaluation framework.

Design Philosophy:
- Google: Statistical rigor, confidence intervals, per-class analysis
- Amazon: Customer impact focus, error cost analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score
)
from typing import Dict, List, Optional, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Rigorous evaluation for both Google and Amazon expectations.
    
    Google: Statistical rigor, confidence intervals, per-class analysis.
    Amazon: Customer impact focus, error cost analysis.
    
    Example:
        >>> evaluator = ModelEvaluator(class_names, error_costs)
        >>> results = evaluator.full_evaluation(y_true, y_pred, y_proba)
    """
    
    def __init__(
        self,
        class_names: List[str],
        error_costs: Optional[Dict[str, float]] = None,
        critical_categories: Optional[List[str]] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names in order
            error_costs: Dict mapping class names to misclassification costs
            critical_categories: List of categories requiring high recall
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Default equal costs
        self.error_costs = error_costs or {name: 1.0 for name in class_names}
        
        # Default critical categories
        self.critical_categories = critical_categories or ['Security', 'Database']
    
    def full_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Comprehensive evaluation report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Basic metrics
        results['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Classification report (Google: detailed per-class analysis)
        results['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Macro and weighted F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        results['per_class_metrics'] = {}
        for i, name in enumerate(self.class_names):
            results['per_class_metrics'][name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Aggregate metrics
        results['macro_precision'] = float(precision.mean())
        results['macro_recall'] = float(recall.mean())
        results['macro_f1'] = float(f1.mean())
        results['weighted_f1'] = float(
            np.average(f1, weights=support) if support.sum() > 0 else 0
        )
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Probability-based metrics (if available)
        if y_proba is not None:
            # AUC-ROC (One-vs-Rest)
            try:
                results['auc_roc_macro'] = float(roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                ))
                results['auc_roc_weighted'] = float(roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                ))
            except ValueError as e:
                logger.warning(f"Could not compute AUC-ROC: {e}")
                results['auc_roc_macro'] = None
                results['auc_roc_weighted'] = None
            
            # Calibration error (Google: confidence meaningfulness)
            results['expected_calibration_error'] = float(
                self._calibration_error(y_true, y_proba)
            )
            
            # Confidence statistics
            confidences = np.max(y_proba, axis=1)
            results['confidence_stats'] = {
                'mean': float(confidences.mean()),
                'std': float(confidences.std()),
                'min': float(confidences.min()),
                'max': float(confidences.max()),
                'low_confidence_rate': float((confidences < 0.7).mean())
            }
        
        # Business impact metrics (Amazon)
        results['weighted_error_cost'] = float(
            self._weighted_error_cost(y_true, y_pred)
        )
        results['critical_category_recall'] = self._critical_category_analysis(
            y_true, y_pred
        )
        
        # Sample counts
        results['total_samples'] = int(len(y_true))
        results['correct_predictions'] = int((y_true == y_pred).sum())
        results['error_count'] = int((y_true != y_pred).sum())
        
        return results
    
    def _calibration_error(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error - how trustworthy are confidence scores?
        
        Google: Critical for understanding if our confidence estimates are reliable.
        """
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        ece = 0.0
        
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = in_bin.sum()
            
            if bin_size > 0:
                bin_accuracy = accuracies[in_bin].mean()
                bin_confidence = confidences[in_bin].mean()
                ece += (bin_size / len(y_true)) * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _weighted_error_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate business-weighted error cost.
        
        Amazon: Not all errors are equal - security misclassification costs more.
        """
        total_cost = 0.0
        error_count = 0
        
        for true_idx, pred_idx in zip(y_true, y_pred):
            if true_idx != pred_idx:
                true_class = self.class_names[true_idx]
                cost = self.error_costs.get(true_class, 1.0)
                total_cost += cost
                error_count += 1
        
        return total_cost / max(len(y_true), 1)
    
    def _critical_category_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Recall analysis on critical categories.
        
        Amazon: Some categories MUST not be missed (Security, Outages).
        """
        results = {}
        
        for cat in self.critical_categories:
            if cat in self.class_names:
                idx = self.class_names.index(cat)
                mask = y_true == idx
                
                if mask.sum() > 0:
                    recall = float((y_pred[mask] == idx).mean())
                    results[cat] = {
                        'recall': recall,
                        'support': int(mask.sum()),
                        'missed': int((y_pred[mask] != idx).sum())
                    }
        
        return results
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            output_path: Optional path to save the report
            
        Returns:
            Report as a string
        """
        results = self.full_evaluation(y_true, y_pred, y_proba)
        
        report_lines = [
            "=" * 60,
            "TICKET CLASSIFIER EVALUATION REPORT",
            "=" * 60,
            "",
            f"Total Samples: {results['total_samples']}",
            f"Correct: {results['correct_predictions']} ({results['accuracy']*100:.1f}%)",
            f"Errors: {results['error_count']}",
            "",
            "-" * 40,
            "AGGREGATE METRICS",
            "-" * 40,
            f"Macro Precision: {results['macro_precision']:.4f}",
            f"Macro Recall: {results['macro_recall']:.4f}",
            f"Macro F1: {results['macro_f1']:.4f}",
            f"Weighted F1: {results['weighted_f1']:.4f}",
        ]
        
        if results.get('auc_roc_macro'):
            report_lines.append(f"AUC-ROC (Macro): {results['auc_roc_macro']:.4f}")
        
        if results.get('expected_calibration_error') is not None:
            report_lines.append(
                f"Calibration Error: {results['expected_calibration_error']:.4f}"
            )
        
        # Critical categories
        if results.get('critical_category_recall'):
            report_lines.extend([
                "",
                "-" * 40,
                "CRITICAL CATEGORY ANALYSIS (Amazon Focus)",
                "-" * 40,
            ])
            for cat, metrics in results['critical_category_recall'].items():
                report_lines.append(
                    f"{cat}: Recall={metrics['recall']:.2%}, "
                    f"Missed={metrics['missed']}/{metrics['support']}"
                )
        
        # Business cost
        report_lines.extend([
            "",
            f"Weighted Error Cost: {results['weighted_error_cost']:.4f}",
        ])
        
        # Per-class breakdown
        report_lines.extend([
            "",
            "-" * 40,
            "PER-CLASS BREAKDOWN",
            "-" * 40,
        ])
        
        for name, metrics in results['per_class_metrics'].items():
            report_lines.append(
                f"{name:20s} | P: {metrics['precision']:.3f} | "
                f"R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | "
                f"N: {metrics['support']}"
            )
        
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def save_results(
        self,
        results: Dict,
        output_path: str
    ):
        """Save evaluation results as JSON."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
