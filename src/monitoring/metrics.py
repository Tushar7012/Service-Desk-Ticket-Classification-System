"""
Prometheus metrics for production monitoring.

Design Philosophy:
- Google: Detailed performance metrics for analysis
- Amazon: Operational metrics tied to SLAs
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Prediction metrics
PREDICTION_COUNTER = Counter(
    'ticket_predictions_total',
    'Total number of predictions made',
    ['category', 'model_version', 'requires_review']
)

PREDICTION_LATENCY = Histogram(
    'ticket_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.5]
)

CONFIDENCE_HISTOGRAM = Histogram(
    'ticket_prediction_confidence',
    'Distribution of prediction confidences',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

LOW_CONFIDENCE_COUNTER = Counter(
    'ticket_low_confidence_predictions_total',
    'Predictions below confidence threshold',
    ['category']
)

ERROR_COUNTER = Counter(
    'ticket_prediction_errors_total',
    'Total prediction errors',
    ['error_type']
)

# Model info
MODEL_INFO = Info(
    'ticket_classifier_model',
    'Information about the loaded model'
)

# Current state gauges
REQUESTS_IN_FLIGHT = Gauge(
    'ticket_requests_in_flight',
    'Number of requests currently being processed'
)

MODEL_LOADED = Gauge(
    'ticket_model_loaded',
    'Whether the model is loaded (1) or not (0)'
)


class MetricsCollector:
    """
    Collect and expose metrics for monitoring dashboards.
    
    Google: Detailed performance metrics for analysis.
    Amazon: Operational metrics tied to SLAs.
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_prediction("Network", 0.92, 0.045, "v1.0")
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize metrics collector.
        
        Args:
            confidence_threshold: Threshold for low confidence flag
        """
        self.confidence_threshold = confidence_threshold
        MODEL_LOADED.set(0)
    
    def set_model_info(self, model_version: str, model_name: str):
        """Set model information for monitoring."""
        MODEL_INFO.info({
            'version': model_version,
            'name': model_name,
            'loaded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        })
        MODEL_LOADED.set(1)
        logger.info(f"Model metrics initialized: {model_name} {model_version}")
    
    def record_prediction(
        self,
        category: str,
        confidence: float,
        latency_seconds: float,
        model_version: str
    ):
        """
        Record a successful prediction.
        
        Args:
            category: Predicted category
            confidence: Prediction confidence
            latency_seconds: Inference latency
            model_version: Model version used
        """
        requires_review = str(confidence < self.confidence_threshold).lower()
        
        # Increment prediction counter
        PREDICTION_COUNTER.labels(
            category=category,
            model_version=model_version,
            requires_review=requires_review
        ).inc()
        
        # Record latency
        PREDICTION_LATENCY.observe(latency_seconds)
        
        # Record confidence
        CONFIDENCE_HISTOGRAM.observe(confidence)
        
        # Track low confidence
        if confidence < self.confidence_threshold:
            LOW_CONFIDENCE_COUNTER.labels(category=category).inc()
    
    def record_error(self, error_type: str):
        """Record a prediction error."""
        ERROR_COUNTER.labels(error_type=error_type).inc()
    
    def request_started(self):
        """Mark that a request has started."""
        REQUESTS_IN_FLIGHT.inc()
    
    def request_completed(self):
        """Mark that a request has completed."""
        REQUESTS_IN_FLIGHT.dec()


class LatencyTracker:
    """Context manager for tracking request latency."""
    
    def __init__(self, histogram: Histogram = PREDICTION_LATENCY):
        self.histogram = histogram
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            self.histogram.observe(duration)
        return False
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time:
            return (time.perf_counter() - self.start_time) * 1000
        return 0.0
