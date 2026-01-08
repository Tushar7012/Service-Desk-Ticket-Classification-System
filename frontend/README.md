# Ticket Classifier AI - Frontend

A futuristic web interface for the AI-powered ticket classification system.

## Features

- Dark theme with animated green diagonal lines
- Real-time ticket classification
- Confidence scores with visual indicators
- Auto-routing status display
- Mobile responsive design

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

Then open: http://localhost:5000

## Deploy to Render

### Option 1: Deploy with render.yaml (Recommended)

1. Push this `frontend/` folder to a GitHub repository
2. Connect to Render.com
3. Select "New Web Service"
4. Connect your GitHub repo
5. Render will detect `render.yaml` automatically
6. Upload your trained model `ticket_classifier.pt` to the service

### Option 2: Manual Deploy

1. Create new Web Service on Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn app:app`
4. Set Python version: 3.10

## Model File

After training in Colab, download `ticket_classifier.pt` and place it in this directory.

## API Endpoints

- `GET /` - Main UI
- `POST /api/predict` - Classify ticket
- `GET /api/health` - Health check
- `GET /api/categories` - List categories

## Tech Stack

- Flask (Python backend)
- Vanilla JS (Frontend)
- DistilBERT (ML model)
- Gunicorn (Production server)
