"""
Evaluation script for ticket classifier.

Usage:
    python scripts/evaluate.py --model-path models/production/model.pt
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TicketClassifier
from src.evaluation import ModelEvaluator, ErrorAnalyzer
from src.preprocessing import TicketPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ticket classifier")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/production.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to test data (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory for output files"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = config['class_names']
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = TicketClassifier(
        num_classes=len(class_names),
        model_name=config['model']['name']
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load tokenizer and preprocessor
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    preprocessor = TicketPreprocessor()
    
    # Load test data
    test_path = args.test_path or "data/test.csv"
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Create label mapping
    label_mapping = {name: i for i, name in enumerate(class_names)}
    
    # Run inference
    logger.info("Running inference...")
    all_preds = []
    all_proba = []
    all_labels = []
    all_texts = []
    
    max_length = config['model'].get('max_length', 256)
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            # Preprocess
            text = preprocessor.combine_fields(
                str(row.get('subject', '')),
                str(row.get('description', ''))
            )
            all_texts.append(text)
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding='max_length'
            ).to(device)
            
            # Predict
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            
            all_proba.append(proba)
            all_preds.append(np.argmax(proba))
            
            # Get label
            label_name = row.get('category', row.get('label', class_names[0]))
            all_labels.append(label_mapping.get(label_name, 0))
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)
    
    # Evaluate
    logger.info("Computing metrics...")
    evaluator = ModelEvaluator(
        class_names=class_names,
        critical_categories=['Security', 'Database']
    )
    
    results = evaluator.full_evaluation(y_true, y_pred, y_proba)
    
    # Save results
    evaluator.save_results(results, output_dir / "metrics.json")
    
    # Generate report
    report = evaluator.generate_report(y_true, y_pred, y_proba)
    print(report)
    (output_dir / "report.txt").write_text(report)
    
    # Error analysis
    logger.info("Analyzing errors...")
    analyzer = ErrorAnalyzer(class_names)
    error_df = analyzer.analyze_errors(all_texts, y_true, y_pred, y_proba)
    error_df.to_csv(output_dir / "errors.csv", index=False)
    
    error_report = analyzer.generate_error_report(all_texts, y_true, y_pred, y_proba)
    print(error_report)
    (output_dir / "error_report.txt").write_text(error_report)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
