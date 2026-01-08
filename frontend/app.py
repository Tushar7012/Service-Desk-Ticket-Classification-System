"""
Flask API for Ticket Classification - Production Ready for Render
Downloads model from Hugging Face Hub on startup.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import random

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
HF_REPO_ID = os.environ.get('HF_REPO_ID', 'Tushar7012/ticket-classifier')
MODEL_FILENAME = 'ticket_classifier.pt'

# Try to load model, fallback to demo mode
MODEL_LOADED = False
model = None
tokenizer = None
preprocessor = None
device = "cpu"

CLASS_NAMES = [
    "Access Management", "Backup", "Database", "Email", 
    "General Inquiry", "Hardware", "Network", "Other",
    "Printing", "Security", "Software", "Storage"
]

# Keywords for demo mode classification
KEYWORDS = {
    "Hardware": ["laptop", "screen", "keyboard", "mouse", "battery", "monitor", "printer", "overheating", "fan", "usb"],
    "Software": ["install", "update", "crash", "freeze", "application", "app", "office", "adobe", "license", "compatible"],
    "Network": ["wifi", "internet", "vpn", "network", "connection", "disconnect", "slow", "ping", "ethernet", "firewall"],
    "Security": ["suspicious", "phishing", "malware", "virus", "breach", "hack", "unauthorized", "stolen", "ransomware", "password"],
    "Access Management": ["access", "permission", "login", "locked", "mfa", "authentication", "sharepoint", "jira", "github"],
    "Email": ["email", "outlook", "calendar", "mailbox", "spam", "attachment", "signature", "sync"],
    "Database": ["sql", "database", "query", "oracle", "backup", "restore", "connection", "timeout", "storage"],
    "Storage": ["onedrive", "storage", "drive", "files", "quota", "sync", "sharepoint", "upload"],
    "Printing": ["print", "printer", "scan", "queue", "paper", "toner", "driver"],
    "Backup": ["backup", "restore", "recovery", "tape", "incremental", "disaster"],
    "General Inquiry": ["how", "what", "where", "when", "policy", "process", "help", "question"],
    "Other": []
}

def demo_predict(text):
    """Demo prediction based on keywords - used when model not loaded."""
    text_lower = text.lower()
    scores = {}
    
    for category, keywords in KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score + random.uniform(0, 0.5)
    
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    total = sum(s for _, s in sorted_cats[:3]) + 0.1
    return [
        {"category": cat, "confidence": (score / total) * 100}
        for cat, score in sorted_cats[:3]
    ]


def download_model_from_hf():
    """Download model from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading model from Hugging Face: {HF_REPO_ID}")
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir="./model_cache"
        )
        print(f"Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Failed to download from HF: {e}")
        return None


def load_model():
    """Load model from local file or download from HF."""
    global model, tokenizer, preprocessor, device, MODEL_LOADED
    
    try:
        import torch
        import torch.nn as nn
        from transformers import DistilBertModel, AutoTokenizer
        import re
        
        class TicketClassifier(nn.Module):
            def __init__(self, num_classes, model_name="distilbert-base-uncased", dropout=0.3):
                super().__init__()
                self.bert = DistilBertModel.from_pretrained(model_name)
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(768, 256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                return self.classifier(outputs.last_hidden_state[:, 0, :])
            
            def predict_proba(self, input_ids, attention_mask):
                logits = self.forward(input_ids, attention_mask)
                return torch.softmax(logits, dim=-1)
        
        class TicketPreprocessor:
            def __init__(self):
                self._email = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')
            def clean(self, text):
                return ' '.join(self._email.sub('[EMAIL]', str(text or '')).lower().split())
            def combine(self, subj, desc):
                return f"[SUBJECT] {self.clean(subj)} [SEP] [DESCRIPTION] {self.clean(desc)}"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        preprocessor = TicketPreprocessor()
        
        # Try local file first
        model_path = os.environ.get('MODEL_PATH', 'ticket_classifier.pt')
        
        if not os.path.exists(model_path):
            # Try to download from Hugging Face
            model_path = download_model_from_hf()
        
        if model_path and os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = TicketClassifier(num_classes=len(CLASS_NAMES))
            
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            MODEL_LOADED = True
            print(f"Model loaded successfully from {model_path}")
        else:
            print("No model available, using demo mode")
            
    except Exception as e:
        print(f"Could not load ML libraries: {e}")
        print("Running in demo mode with keyword-based classification")


# Load model on startup
load_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        subject = data.get('subject', '')
        description = data.get('description', '')
        
        if not subject and not description:
            return jsonify({'error': 'Subject or description required'}), 400
        
        text = f"{subject} {description}"
        
        if MODEL_LOADED and model is not None:
            combined = preprocessor.combine(subject, description)
            inputs = tokenizer(
                combined,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding='max_length'
            ).to(device)
            
            import torch
            with torch.no_grad():
                probs = model.predict_proba(inputs['input_ids'], inputs['attention_mask'])[0]
            
            probs_np = probs.cpu().numpy()
            top_3_idx = probs_np.argsort()[-3:][::-1]
            
            predictions = [
                {"category": CLASS_NAMES[idx], "confidence": float(probs_np[idx]) * 100}
                for idx in top_3_idx
            ]
        else:
            predictions = demo_predict(text)
        
        requires_review = predictions[0]['confidence'] < 70
        
        return jsonify({
            'success': True,
            'primary': predictions[0],
            'alternatives': predictions[1:],
            'requires_review': requires_review,
            'demo_mode': not MODEL_LOADED
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'demo_mode': not MODEL_LOADED,
        'hf_repo': HF_REPO_ID
    })


@app.route('/api/categories')
def categories():
    return jsonify({'categories': CLASS_NAMES})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*50}")
    print(f"  TICKET CLASSIFIER AI")
    print(f"  Mode: {'Production' if MODEL_LOADED else 'Demo (keyword-based)'}")
    print(f"  HF Repo: {HF_REPO_ID}")
    print(f"  URL: http://localhost:{port}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
