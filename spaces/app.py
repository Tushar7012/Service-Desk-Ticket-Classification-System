"""
IT Ticket Classifier - HuggingFace Spaces App
Gradio interface for classifying IT support tickets
"""

import gradio as gr
import torch
import torch.nn as nn
from transformers import DistilBertModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import re
import os

# Configuration
HF_REPO_ID = "TuShar2309/ticket-classifier"
MODEL_FILENAME = "ticket_classifier.pt"

CLASS_NAMES = [
    "Access Management", "Backup", "Database", "Email", 
    "General Inquiry", "Hardware", "Network", "Other",
    "Printing", "Security", "Software", "Storage"
]

# Category descriptions for display
CATEGORY_INFO = {
    "Access Management": "🔐 Login, permissions, MFA, account issues",
    "Backup": "💾 Backup and restore operations",
    "Database": "🗄️ SQL, database connectivity, queries",
    "Email": "📧 Outlook, calendar, mailbox issues",
    "General Inquiry": "❓ How-to questions, policies",
    "Hardware": "💻 Laptop, monitor, keyboard, mouse",
    "Network": "🌐 WiFi, VPN, internet connectivity",
    "Other": "📋 Miscellaneous requests",
    "Printing": "🖨️ Printers, scanning, print queue",
    "Security": "🔒 Threats, malware, security incidents",
    "Software": "📦 Application issues, installations",
    "Storage": "📁 OneDrive, SharePoint, file storage"
}


class TicketPreprocessor:
    def __init__(self):
        self._email = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')
    
    def clean(self, text):
        return ' '.join(self._email.sub('[EMAIL]', str(text or '')).lower().split())
    
    def combine(self, subject, description):
        return f"[SUBJECT] {self.clean(subject)} [SEP] [DESCRIPTION] {self.clean(description)}"


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


# Load model
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

try:
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    print(f"Model downloaded: {model_path}")
    
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
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False

preprocessor = TicketPreprocessor()


def classify_ticket(subject, description):
    """Classify a ticket and return results."""
    if not subject and not description:
        return "⚠️ Please enter a subject or description", "", ""
    
    if not MODEL_LOADED:
        return "❌ Model not loaded", "", ""
    
    try:
        # Preprocess and tokenize
        combined = preprocessor.combine(subject, description)
        inputs = tokenizer(
            combined,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding='max_length'
        ).to(device)
        
        # Predict
        with torch.no_grad():
            probs = model.predict_proba(inputs['input_ids'], inputs['attention_mask'])[0]
        
        probs_np = probs.cpu().numpy()
        top_indices = probs_np.argsort()[::-1]
        
        # Primary prediction
        primary_idx = top_indices[0]
        primary_cat = CLASS_NAMES[primary_idx]
        primary_conf = probs_np[primary_idx] * 100
        
        # Status
        if primary_conf >= 80:
            status = "✅ **High Confidence** - Auto-route recommended"
        elif primary_conf >= 60:
            status = "⚠️ **Medium Confidence** - Review suggested"
        else:
            status = "🔍 **Low Confidence** - Human review required"
        
        # Format primary result
        primary_result = f"""
## {CATEGORY_INFO.get(primary_cat, primary_cat)}

### Predicted Category: **{primary_cat}**
### Confidence: **{primary_conf:.1f}%**

{status}
"""
        
        # Format alternatives
        alternatives = "### Other Possibilities:\n\n"
        for i in range(1, min(4, len(top_indices))):
            idx = top_indices[i]
            cat = CLASS_NAMES[idx]
            conf = probs_np[idx] * 100
            alternatives += f"- **{cat}**: {conf:.1f}%\n"
        
        # Confidence bar
        conf_display = f"{'█' * int(primary_conf / 5)}{'░' * (20 - int(primary_conf / 5))} {primary_conf:.1f}%"
        
        return primary_result, alternatives, conf_display
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


# Example tickets
examples = [
    ["VPN not connecting", "Cannot connect to corporate VPN from home, getting timeout error"],
    ["Suspicious email received", "Got an email asking for my password, looks like phishing"],
    ["Need SharePoint access", "Just joined the marketing team, need access to the team SharePoint"],
    ["Laptop screen flickering", "My laptop screen has been flickering intermittently since yesterday"],
    ["Outlook not receiving emails", "Haven't received any emails in Outlook for the past 3 hours"],
    ["How to reset password", "What is the process to reset my Active Directory password?"],
    ["Printer not working", "Print jobs stuck in queue and won't print"],
    ["SQL query slow", "Database query that used to take 2 seconds now takes 10 minutes"],
]


# Create Gradio interface
with gr.Blocks(
    title="IT Ticket Classifier",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"),
    css="""
    .gradio-container { max-width: 900px !important; }
    .primary-result { font-size: 1.2em; }
    """
) as demo:
    gr.Markdown("""
    # 🎫 IT Service Desk Ticket Classifier
    
    **Powered by DistilBERT** | Classifies tickets into 12 IT support categories
    
    Enter a ticket subject and description below to get the predicted category.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            subject_input = gr.Textbox(
                label="📋 Ticket Subject",
                placeholder="e.g., VPN not connecting",
                lines=1
            )
            description_input = gr.Textbox(
                label="📝 Ticket Description",
                placeholder="e.g., Cannot connect to corporate VPN from home, getting timeout error after 30 seconds...",
                lines=4
            )
            classify_btn = gr.Button("🔍 Classify Ticket", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            primary_output = gr.Markdown(label="Primary Prediction")
            confidence_output = gr.Textbox(label="Confidence", interactive=False)
            alternatives_output = gr.Markdown(label="Alternatives")
    
    classify_btn.click(
        fn=classify_ticket,
        inputs=[subject_input, description_input],
        outputs=[primary_output, alternatives_output, confidence_output]
    )
    
    gr.Examples(
        examples=examples,
        inputs=[subject_input, description_input],
        outputs=[primary_output, alternatives_output, confidence_output],
        fn=classify_ticket,
        cache_examples=False
    )
    
    gr.Markdown("""
    ---
    ### 📊 Supported Categories
    
    | Category | Description |
    |----------|-------------|
    | Access Management | Login, permissions, MFA |
    | Backup | Backup and restore |
    | Database | SQL, queries, DB issues |
    | Email | Outlook, calendar |
    | General Inquiry | How-to questions |
    | Hardware | Devices, laptops |
    | Network | WiFi, VPN, internet |
    | Other | Miscellaneous |
    | Printing | Printers, scanning |
    | Security | Threats, incidents |
    | Software | Applications |
    | Storage | OneDrive, SharePoint |
    
    ---
    **Model**: DistilBERT fine-tuned on 5,760 IT support tickets
    """)


if __name__ == "__main__":
    demo.launch()
