"""
Production REST API for ticket classification.

Design Philosophy:
- Google: Clean abstraction between model and serving layer
- Amazon: Reliability, latency monitoring, graceful degradation
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer
import logging
import time
from datetime import datetime
from typing import Optional
from pathlib import Path
import yaml

from src.api.schemas import (
    TicketRequest,
    PredictionResponse,
    PredictionDetail,
    BatchTicketRequest,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse
)
from src.preprocessing import TicketPreprocessor
from src.models import TicketClassifier

logger = logging.getLogger(__name__)

# Global state
_service: Optional['InferenceService'] = None
_start_time: Optional[float] = None


class InferenceService:
    """
    Production inference service with proper error handling.
    
    Google: Clean abstraction between model and serving layer.
    Amazon: Reliability, latency monitoring, graceful degradation.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cuda"
    ):
        """
        Initialize inference service.
        
        Args:
            model_path: Path to model checkpoint
            config_path: Path to configuration file
            device: Device for inference
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config.get('class_names', [])
        self.confidence_threshold = self.config.get('inference', {}).get(
            'confidence_threshold', 0.7
        )
        self.max_length = self.config.get('model', {}).get('max_length', 256)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        model_name = self.config.get('model', {}).get('name', 'distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize preprocessor
        self.preprocessor = TicketPreprocessor()
        
        # Model version
        self.model_version = self._get_model_version(model_path)
        
        logger.info(f"Service ready. Device: {self.device}, Version: {self.model_version}")
    
    def _load_model(self, model_path: str) -> TicketClassifier:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        num_classes = len(self.class_names)
        model_name = self.config.get('model', {}).get('name', 'distilbert-base-uncased')
        
        model = TicketClassifier(
            num_classes=num_classes,
            model_name=model_name
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _get_model_version(self, model_path: str) -> str:
        """Extract or generate model version."""
        path = Path(model_path)
        
        # Try to get from filename or config
        if 'v' in path.stem:
            return path.stem
        
        # Use modification time as version
        mtime = path.stat().st_mtime if path.exists() else time.time()
        return f"v{datetime.fromtimestamp(mtime).strftime('%Y%m%d_%H%M')}"
    
    def predict(self, subject: str, description: str) -> dict:
        """
        Predict category for a single ticket.
        
        Args:
            subject: Ticket subject
            description: Ticket description
            
        Returns:
            Prediction dictionary
        """
        start_time = time.perf_counter()
        
        try:
            # Preprocess
            text = self.preprocessor.combine_fields(subject, description)
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding='max_length'
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                proba = torch.softmax(logits, dim=-1)[0]
            
            # Get predictions
            proba_np = proba.cpu().numpy()
            top_3_idx = proba_np.argsort()[-3:][::-1]
            
            top_3 = [
                PredictionDetail(
                    category=self.class_names[idx],
                    confidence=float(proba_np[idx])
                )
                for idx in top_3_idx
            ]
            
            confidence = float(proba_np[top_3_idx[0]])
            requires_review = confidence < self.confidence_threshold
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return {
                "category": self.class_names[top_3_idx[0]],
                "confidence": confidence,
                "top_3_predictions": top_3,
                "requires_review": requires_review,
                "model_version": self.model_version,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _service, _start_time
    
    _start_time = time.time()
    
    # Try to load model on startup
    try:
        config_path = Path("configs/production.yaml")
        model_path = Path("models/production/model.pt")
        
        if config_path.exists() and model_path.exists():
            _service = InferenceService(
                model_path=str(model_path),
                config_path=str(config_path)
            )
            logger.info("Model loaded successfully on startup")
        else:
            logger.warning(
                "Model/config not found. API will start but predictions disabled."
            )
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Ticket Classification API",
    description="Production API for automatic service desk ticket classification",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> InferenceService:
    """Dependency to get inference service."""
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    return _service


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a single ticket",
    tags=["Prediction"]
)
async def predict(
    request: TicketRequest,
    service: InferenceService = Depends(get_service)
) -> PredictionResponse:
    """
    Classify a service desk ticket.
    
    Returns predicted category with confidence and top-3 alternatives.
    """
    try:
        result = service.predict(request.subject, request.description)
        
        return PredictionResponse(
            request_id=request.request_id or f"req_{int(time.time()*1000)}",
            **result
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Classify multiple tickets",
    tags=["Prediction"]
)
async def predict_batch(
    request: BatchTicketRequest,
    service: InferenceService = Depends(get_service)
) -> BatchPredictionResponse:
    """Batch classification for multiple tickets."""
    start_time = time.perf_counter()
    
    predictions = []
    for ticket in request.tickets:
        try:
            result = service.predict(ticket.subject, ticket.description)
            pred = PredictionResponse(
                request_id=ticket.request_id or f"req_{int(time.time()*1000)}",
                **result
            )
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    total_latency = (time.perf_counter() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_latency_ms=total_latency,
        batch_size=len(predictions)
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Health"]
)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancer."""
    uptime = time.time() - _start_time if _start_time else None
    
    return HealthResponse(
        status="healthy" if _service else "degraded",
        model_loaded=_service is not None,
        model_version=_service.model_version if _service else None,
        uptime_seconds=uptime
    )


@app.get(
    "/ready",
    summary="Readiness check",
    tags=["Health"]
)
async def readiness_check():
    """Kubernetes readiness probe."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}
