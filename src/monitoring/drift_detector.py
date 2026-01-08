"""
Production monitoring for data and prediction drift.

Design Philosophy:
- Google: Statistical rigor in drift detection (KS-test, PSI)
- Amazon: Alarms tied to business metrics and SLAs
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from collections import deque
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Production monitoring for data and prediction drift.
    
    Google: Statistical rigor in drift detection (PSI, KS-test).
    Amazon: Alarms tied to business metrics and SLAs.
    
    Example:
        >>> detector = DriftDetector(reference_dist, class_names)
        >>> detector.add_prediction(pred_idx, confidence)
        >>> alerts = detector.check_all()
    """
    
    def __init__(
        self,
        reference_distribution: np.ndarray,
        class_names: List[str],
        psi_threshold: float = 0.2,
        confidence_threshold: float = 0.7,
        window_size: int = 1000
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_distribution: Training data class distribution
            class_names: List of class names
            psi_threshold: PSI threshold for drift alert
            confidence_threshold: Below this = low confidence
            window_size: Rolling window size for monitoring
        """
        self.reference_distribution = reference_distribution
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.psi_threshold = psi_threshold
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        
        # Rolling windows for real-time monitoring
        self.confidence_window = deque(maxlen=window_size)
        self.prediction_window = deque(maxlen=window_size)
        self.timestamp_window = deque(maxlen=window_size)
        
        # Alert history
        self.alert_history = []
    
    def add_prediction(
        self,
        prediction: int,
        confidence: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Add a prediction to the monitoring window.
        
        Args:
            prediction: Predicted class index
            confidence: Prediction confidence
            timestamp: Prediction timestamp
        """
        self.confidence_window.append(confidence)
        self.prediction_window.append(prediction)
        self.timestamp_window.append(timestamp or datetime.utcnow())
    
    def check_confidence_drift(self) -> Dict:
        """
        Monitor confidence score distribution.
        
        Amazon: Low confidence = uncertain predictions = customer risk.
        
        Returns:
            Dict with confidence metrics and alert status
        """
        if len(self.confidence_window) < 100:
            return {"status": "insufficient_data", "count": len(self.confidence_window)}
        
        confidences = np.array(self.confidence_window)
        low_conf_rate = (confidences < self.confidence_threshold).mean()
        
        # Alert if >15% predictions are low confidence
        alert = low_conf_rate > 0.15
        
        result = {
            "mean_confidence": float(confidences.mean()),
            "std_confidence": float(confidences.std()),
            "low_confidence_rate": float(low_conf_rate),
            "threshold": self.confidence_threshold,
            "alert": alert,
            "window_size": len(confidences)
        }
        
        if alert:
            self._record_alert("confidence_drift", result)
        
        return result
    
    def check_prediction_drift(self) -> Dict:
        """
        Check if prediction distribution has shifted from training.
        
        Google: Population Stability Index for distribution comparison.
        
        Returns:
            Dict with PSI value and alert status
        """
        if len(self.prediction_window) < self.window_size:
            return {"status": "insufficient_data", "count": len(self.prediction_window)}
        
        # Calculate current distribution
        predictions = np.array(self.prediction_window)
        current_counts = np.bincount(predictions, minlength=self.num_classes)
        current_dist = current_counts / current_counts.sum()
        
        # Calculate PSI
        psi = self._calculate_psi(self.reference_distribution, current_dist)
        
        # Determine drift level
        if psi < 0.1:
            drift_level = "no_drift"
        elif psi < 0.2:
            drift_level = "minor_drift"
        else:
            drift_level = "significant_drift"
        
        alert = psi > self.psi_threshold
        
        result = {
            "psi": float(psi),
            "drift_level": drift_level,
            "alert": alert,
            "current_distribution": current_dist.tolist(),
            "reference_distribution": self.reference_distribution.tolist(),
            "window_size": len(predictions)
        }
        
        if alert:
            self._record_alert("prediction_drift", result)
        
        return result
    
    def check_class_imbalance_shift(self) -> Dict:
        """
        Check if class distribution has shifted significantly.
        
        Useful for detecting changes in ticket patterns (e.g., after a system update).
        """
        if len(self.prediction_window) < self.window_size:
            return {"status": "insufficient_data"}
        
        predictions = np.array(self.prediction_window)
        current_counts = np.bincount(predictions, minlength=self.num_classes)
        
        # Find classes with significant changes
        expected_counts = self.reference_distribution * len(predictions)
        
        significant_changes = {}
        for i, name in enumerate(self.class_names):
            if expected_counts[i] > 0:
                ratio = current_counts[i] / expected_counts[i]
                if ratio > 1.5 or ratio < 0.5:  # 50% change threshold
                    significant_changes[name] = {
                        "expected_ratio": float(self.reference_distribution[i]),
                        "current_ratio": float(current_counts[i] / len(predictions)),
                        "change_factor": float(ratio)
                    }
        
        return {
            "significant_changes": significant_changes,
            "alert": len(significant_changes) > 0
        }
    
    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change, investigate
        PSI > 0.2: Significant change, action required
        """
        expected = np.clip(expected, epsilon, 1)
        actual = np.clip(actual, epsilon, 1)
        
        psi = np.sum((actual - expected) * np.log(actual / expected))
        return float(psi)
    
    def _record_alert(self, alert_type: str, details: Dict):
        """Record alert in history."""
        alert = {
            "type": alert_type,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        self.alert_history.append(alert)
        logger.warning(f"ALERT: {alert_type} detected - {details}")
    
    def check_all(self) -> Dict:
        """Run all drift checks and return combined results."""
        return {
            "confidence": self.check_confidence_drift(),
            "prediction_drift": self.check_prediction_drift(),
            "class_imbalance": self.check_class_imbalance_shift(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]).timestamp() > cutoff
        ]


class RetrainingTrigger:
    """
    Automated retraining decision system.
    
    Amazon: Cost-benefit analysis of retraining.
    Google: Statistical evidence for model degradation.
    """
    
    def __init__(
        self,
        min_samples_for_evaluation: int = 5000,
        accuracy_drop_threshold: float = 0.03,
        retraining_cooldown_days: int = 7
    ):
        """
        Initialize retraining trigger.
        
        Args:
            min_samples_for_evaluation: Minimum samples needed
            accuracy_drop_threshold: Accuracy drop to trigger retraining
            retraining_cooldown_days: Days between retraining
        """
        self.min_samples = min_samples_for_evaluation
        self.accuracy_threshold = accuracy_drop_threshold
        self.cooldown_days = retraining_cooldown_days
        self.last_retrain_date: Optional[datetime] = None
    
    def should_retrain(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        drift_results: Dict
    ) -> Dict:
        """
        Determine if retraining is needed.
        
        Args:
            current_metrics: Current evaluation metrics
            baseline_metrics: Baseline metrics from training
            drift_results: Results from drift detection
            
        Returns:
            Decision with justification
        """
        reasons = []
        
        # Check cooldown
        if self.last_retrain_date:
            days_since = (datetime.utcnow() - self.last_retrain_date).days
            if days_since < self.cooldown_days:
                return {
                    "should_retrain": False,
                    "reasons": [f"In cooldown period ({days_since}/{self.cooldown_days} days)"],
                    "recommendation": "Wait for cooldown to complete"
                }
        
        # Check accuracy drop
        current_f1 = current_metrics.get('f1_macro', 1.0)
        baseline_f1 = baseline_metrics.get('f1_macro', 0.85)
        f1_drop = baseline_f1 - current_f1
        
        if f1_drop > self.accuracy_threshold:
            reasons.append(
                f"F1 dropped by {f1_drop:.3f} (threshold: {self.accuracy_threshold})"
            )
        
        # Check drift alerts
        if drift_results.get('prediction_drift', {}).get('alert', False):
            psi = drift_results['prediction_drift'].get('psi', 0)
            reasons.append(f"Significant prediction drift detected (PSI={psi:.3f})")
        
        if drift_results.get('confidence', {}).get('alert', False):
            rate = drift_results['confidence'].get('low_confidence_rate', 0)
            reasons.append(f"High low-confidence rate: {rate:.1%}")
        
        should_retrain = len(reasons) > 0
        
        return {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "current_f1": current_f1,
            "baseline_f1": baseline_f1,
            "recommendation": "Schedule retraining pipeline" if should_retrain else "Continue monitoring",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def record_retraining(self):
        """Record that retraining was performed."""
        self.last_retrain_date = datetime.utcnow()
        logger.info(f"Retraining recorded at {self.last_retrain_date}")
