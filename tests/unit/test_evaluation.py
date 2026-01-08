"""
Unit tests for evaluation module.
"""

import pytest
import numpy as np

from src.evaluation import ModelEvaluator, ErrorAnalyzer


class TestModelEvaluator:
    """Tests for ModelEvaluator."""
    
    @pytest.fixture
    def class_names(self):
        return ["Hardware", "Software", "Network", "Security", "Other"]
    
    @pytest.fixture
    def evaluator(self, class_names):
        return ModelEvaluator(
            class_names=class_names,
            error_costs={"Security": 5.0},
            critical_categories=["Security"]
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample prediction data."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 5
        
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = y_true.copy()
        
        # Introduce some errors
        error_indices = np.random.choice(n_samples, 20, replace=False)
        y_pred[error_indices] = np.random.randint(0, n_classes, 20)
        
        # Generate probabilities
        y_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
        
        return y_true, y_pred, y_proba
    
    def test_full_evaluation_returns_all_keys(self, evaluator, sample_data):
        """Test that full evaluation returns all expected keys."""
        y_true, y_pred, y_proba = sample_data
        
        results = evaluator.full_evaluation(y_true, y_pred, y_proba)
        
        expected_keys = [
            'accuracy', 'classification_report', 'per_class_metrics',
            'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1',
            'confusion_matrix', 'total_samples', 'error_count'
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_accuracy_calculation(self, evaluator):
        """Test accuracy calculation."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 0])  # One wrong
        
        results = evaluator.full_evaluation(y_true, y_pred)
        
        assert results['accuracy'] == 0.8
    
    def test_calibration_error(self, evaluator, sample_data):
        """Test calibration error calculation."""
        y_true, y_pred, y_proba = sample_data
        
        results = evaluator.full_evaluation(y_true, y_pred, y_proba)
        
        ece = results.get('expected_calibration_error')
        assert ece is not None
        assert 0 <= ece <= 1
    
    def test_weighted_error_cost(self, evaluator, class_names):
        """Test business-weighted error cost."""
        # All predictions for Security class (index 3) are wrong
        y_true = np.array([3, 3, 3, 0, 0])  # 3 security tickets
        y_pred = np.array([0, 1, 2, 0, 0])  # All security misclassified
        
        results = evaluator.full_evaluation(y_true, y_pred)
        
        # Security has cost 5.0, 3 errors * 5.0 / 5 samples = 3.0
        assert results['weighted_error_cost'] == 3.0
    
    def test_critical_category_recall(self, evaluator):
        """Test critical category analysis."""
        # Security is at index 3
        y_true = np.array([3, 3, 3, 3, 0])  # 4 security tickets
        y_pred = np.array([3, 3, 0, 0, 0])  # 2 correct, 2 missed
        
        results = evaluator.full_evaluation(y_true, y_pred)
        
        security_recall = results['critical_category_recall']['Security']
        assert security_recall['recall'] == 0.5
        assert security_recall['missed'] == 2


class TestErrorAnalyzer:
    """Tests for ErrorAnalyzer."""
    
    @pytest.fixture
    def class_names(self):
        return ["A", "B", "C"]
    
    @pytest.fixture
    def analyzer(self, class_names):
        return ErrorAnalyzer(class_names=class_names)
    
    def test_analyze_errors_returns_dataframe(self, analyzer):
        """Test that analyze_errors returns a DataFrame."""
        texts = ["text1", "text2", "text3"]
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 1, 0])  # 2 errors
        y_proba = np.array([
            [0.2, 0.6, 0.2],
            [0.1, 0.8, 0.1],
            [0.7, 0.1, 0.2]
        ])
        
        df = analyzer.analyze_errors(texts, y_true, y_pred, y_proba)
        
        assert len(df) == 2  # 2 errors
        assert 'text' in df.columns
        assert 'error_type' in df.columns
    
    def test_confusion_pairs(self, analyzer):
        """Test confusion pair detection."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 2, 2])  # A->B 3 times, B->C 2 times
        
        pairs = analyzer.get_confusion_pairs(y_true, y_pred, top_k=2)
        
        assert len(pairs) == 2
        assert pairs[0][2] == 3  # Most common: 3 occurrences
    
    def test_error_categorization(self, analyzer):
        """Test error type categorization."""
        # High confidence error
        error_type = analyzer._categorize_error(
            true_label=0,
            pred_label=1,
            proba=np.array([0.05, 0.95, 0.0])
        )
        assert error_type == "High-confidence error"
        
        # Low confidence error
        error_type = analyzer._categorize_error(
            true_label=0,
            pred_label=1,
            proba=np.array([0.3, 0.4, 0.3])
        )
        assert error_type == "Low-confidence error"
    
    def test_error_summary(self, analyzer):
        """Test error summary generation."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 2])  # 3 errors
        y_proba = np.random.dirichlet([1, 1, 1], 5)
        
        summary = analyzer.get_error_summary(y_true, y_pred, y_proba)
        
        assert summary['total_errors'] == 3
        assert 'error_type_distribution' in summary
        assert 'per_class_error_rates' in summary
