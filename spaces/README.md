---
title: IT Ticket Classifier
emoji: 🎫
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# IT Service Desk Ticket Classifier

Classify IT support tickets into 12 categories using DistilBERT.

## Categories
- Access Management, Backup, Database, Email
- General Inquiry, Hardware, Network, Other
- Printing, Security, Software, Storage

## Model
- **Base**: DistilBERT (distilbert-base-uncased)
- **Training**: 5,760 IT support tickets
- **Architecture**: DistilBERT + Classifier head
