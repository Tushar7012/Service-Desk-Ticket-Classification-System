"""
Model optimization for production deployment.

Design Philosophy:
- Google: Maintains accuracy within acceptable bounds
- Amazon: Reduces latency and infrastructure costs
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Model optimization utilities for production deployment.
    
    Provides quantization, ONNX export, and benchmarking.
    
    Google: Maintains accuracy within acceptable bounds.
    Amazon: Reduces latency and infrastructure costs.
    """
    
    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization for faster CPU inference.
        
        Benefits:
        - 2-4x speedup on CPU
        - <1% accuracy loss typically
        - No calibration data needed
        
        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")
        
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        # Log size reduction
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        quantized_size = sum(
            p.numel() * (1 if p.dtype == torch.qint8 else p.element_size())
            for p in quantized_model.parameters()
        )
        
        logger.info(
            f"Model size: {original_size / 1e6:.1f}MB -> {quantized_size / 1e6:.1f}MB "
            f"({(1 - quantized_size/original_size) * 100:.1f}% reduction)"
        )
        
        return quantized_model
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        tokenizer,
        output_path: str,
        max_length: int = 256,
        opset_version: int = 14
    ) -> str:
        """
        Export model to ONNX format for runtime optimization.
        
        Amazon: ONNX Runtime provides ~2x speedup on both CPU and GPU.
        
        Returns:
            Path to exported model
        """
        import torch.onnx
        
        model.eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = tokenizer(
            "Sample ticket text for export",
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        
        logger.info(f"Exporting to ONNX: {output_path}")
        
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            str(output_path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            },
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        logger.info(f"ONNX model exported successfully")
        return str(output_path)
    
    @staticmethod
    def benchmark_inference(
        model: nn.Module,
        tokenizer,
        num_samples: int = 100,
        max_length: int = 256,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Benchmark inference latency.
        
        Returns:
            Dictionary with latency percentiles
        """
        model.eval()
        model.to(device)
        
        # Create sample input
        sample_text = "My laptop is not connecting to the VPN and I cannot access internal resources. " * 5
        
        inputs = tokenizer(
            sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Warmup
        logger.info("Warming up...")
        for _ in range(10):
            with torch.no_grad():
                _ = model(**inputs)
        
        # Benchmark
        logger.info(f"Running {num_samples} inference calls...")
        latencies = []
        
        for _ in range(num_samples):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies = np.array(latencies)
        
        results = {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p90_ms': float(np.percentile(latencies, 90)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'device': device,
            'num_samples': num_samples
        }
        
        logger.info(
            f"Latency: mean={results['mean_ms']:.2f}ms, "
            f"p95={results['p95_ms']:.2f}ms, p99={results['p99_ms']:.2f}ms"
        )
        
        return results


class ThresholdOptimizer:
    """
    Optimize classification thresholds per business requirements.
    
    Amazon: Different thresholds for different customer impact levels.
    Google: Precision-recall trade-off optimization.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize threshold optimizer.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.thresholds = {name: 0.5 for name in class_names}
    
    def find_threshold_for_recall(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_idx: int,
        target_recall: float = 0.95
    ) -> float:
        """
        Find threshold that achieves target recall for a class.
        
        Args:
            y_true: Ground truth labels
            y_proba: Prediction probabilities
            class_idx: Index of target class
            target_recall: Target recall value
            
        Returns:
            Optimal threshold
        """
        class_proba = y_proba[:, class_idx]
        class_mask = y_true == class_idx
        
        if class_mask.sum() == 0:
            return 0.5
        
        # Sort by probability
        thresholds = np.linspace(0, 1, 100)
        
        for threshold in thresholds:
            predictions = (class_proba >= threshold).astype(int)
            recall = predictions[class_mask].sum() / class_mask.sum()
            
            if recall >= target_recall:
                return float(threshold)
        
        # Return lowest threshold if target not achievable
        return 0.0
    
    def optimize_all_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_recalls: Optional[Dict[str, float]] = None,
        default_target: float = 0.9
    ) -> Dict[str, float]:
        """
        Optimize thresholds for all classes.
        
        Args:
            y_true: Ground truth labels
            y_proba: Prediction probabilities
            target_recalls: Dict mapping class names to target recalls
            default_target: Default target recall
            
        Returns:
            Dict mapping class names to optimal thresholds
        """
        target_recalls = target_recalls or {}
        
        for i, name in enumerate(self.class_names):
            target = target_recalls.get(name, default_target)
            threshold = self.find_threshold_for_recall(
                y_true, y_proba, i, target
            )
            self.thresholds[name] = threshold
            logger.info(f"{name}: threshold={threshold:.3f} for recall>={target:.2f}")
        
        return self.thresholds
    
    def apply_thresholds(
        self,
        y_proba: np.ndarray,
        default_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply class-specific thresholds to predictions.
        
        Returns:
            (predictions, flags) where flags indicate low-confidence
        """
        predictions = np.argmax(y_proba, axis=1)
        confidences = np.max(y_proba, axis=1)
        
        # Flag predictions below class threshold
        flags = np.zeros(len(predictions), dtype=bool)
        
        for i, pred in enumerate(predictions):
            class_name = self.class_names[pred]
            threshold = self.thresholds.get(class_name, default_threshold)
            
            if confidences[i] < threshold:
                flags[i] = True
        
        return predictions, flags


# Convenience functions
def quantize_model(model: nn.Module) -> nn.Module:
    """Convenience function for dynamic quantization."""
    return ModelOptimizer.quantize_dynamic(model)


def export_to_onnx(
    model: nn.Module,
    tokenizer,
    output_path: str,
    **kwargs
) -> str:
    """Convenience function for ONNX export."""
    return ModelOptimizer.export_to_onnx(model, tokenizer, output_path, **kwargs)
