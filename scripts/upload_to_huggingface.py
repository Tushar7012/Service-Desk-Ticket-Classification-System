"""
Script to upload trained model to Hugging Face Hub.
This allows the Render deployment to download the model on startup.

Usage:
    pip install huggingface_hub
    huggingface-cli login  # Login with your HF token
    python scripts/upload_to_huggingface.py
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_file

def upload_model(
    model_path: str,
    repo_name: str,
    username: str = None,
    private: bool = False
):
    """Upload model checkpoint to Hugging Face Hub."""
    
    api = HfApi()
    
    # Get username if not provided
    if username is None:
        user_info = api.whoami()
        username = user_info["name"]
    
    repo_id = f"{username}/{repo_name}"
    
    print(f"Uploading to: https://huggingface.co/{repo_id}")
    
    # Create repo if doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload model file
    print(f"Uploading {model_path}...")
    
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo="ticket_classifier.pt",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"\nModel uploaded successfully!")
    print(f"URL: https://huggingface.co/{repo_id}")
    print(f"\nTo download in your app, use:")
    print(f'  from huggingface_hub import hf_hub_download')
    print(f'  model_path = hf_hub_download(repo_id="{repo_id}", filename="ticket_classifier.pt")')
    
    return repo_id


def create_model_card(repo_id: str):
    """Create a model card README for the repo."""
    
    model_card = """---
license: mit
language:
- en
tags:
- ticket-classification
- it-support
- distilbert
- text-classification
datasets:
- custom
metrics:
- f1
- accuracy
---

# IT Service Desk Ticket Classifier

A DistilBERT-based model for classifying IT support tickets into 12 categories.

## Model Description

This model classifies IT service desk tickets into the following categories:

| Category | Description |
|----------|-------------|
| Access Management | Login, permissions, MFA issues |
| Backup | Backup and restore operations |
| Database | SQL, database connectivity |
| Email | Outlook, calendar, mailbox |
| General Inquiry | How-to questions |
| Hardware | Physical device issues |
| Network | WiFi, VPN, connectivity |
| Other | Miscellaneous |
| Printing | Printer and scanning |
| Security | Threats, breaches, security |
| Software | Application issues |
| Storage | OneDrive, SharePoint, drives |

## Training Details

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Training Data**: 5,760 IT support tickets
- **Loss Function**: Focal Loss (for class imbalance)
- **Framework**: PyTorch + Transformers

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/ticket-classifier",
    filename="ticket_classifier.pt"
)

# Load model
checkpoint = torch.load(model_path, map_location="cpu")
```

## Performance

Achieves 85%+ macro F1 score across 12 categories.

## License

MIT
"""
    
    api = HfApi()
    
    # Upload README
    from huggingface_hub import upload_file
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(model_card)
        temp_path = f.name
    
    upload_file(
        path_or_fileobj=temp_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    
    os.unlink(temp_path)
    print("Model card uploaded!")


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--model-path",
        type=str,
        default="output/ticket_classifier.pt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="ticket-classifier",
        help="Name for the HF repository"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first or provide correct path.")
        return
    
    print("=" * 60)
    print("Uploading Model to Hugging Face Hub")
    print("=" * 60)
    
    repo_id = upload_model(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private
    )
    
    # Create model card
    print("\nCreating model card...")
    create_model_card(repo_id)
    
    print("\n" + "=" * 60)
    print("Upload complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
