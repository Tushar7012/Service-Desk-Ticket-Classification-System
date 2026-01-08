"""
API request/response schemas.

Uses Pydantic for validation and OpenAPI documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class TicketRequest(BaseModel):
    """Request schema for ticket classification."""
    
    subject: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Ticket subject line"
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Ticket description/body"
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional request ID for tracing"
    )
    priority: Optional[str] = Field(
        None,
        description="Optional priority level (P1-P4)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "subject": "VPN Connection Issues",
                "description": "I am unable to connect to the corporate VPN from home. The client shows 'Connection Timeout' error.",
                "request_id": "REQ-2024-001234",
                "priority": "P2"
            }
        }


class PredictionDetail(BaseModel):
    """Single prediction with category and confidence."""
    
    category: str = Field(..., description="Category name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class PredictionResponse(BaseModel):
    """Response schema for ticket classification."""
    
    request_id: str = Field(..., description="Request ID for tracing")
    category: str = Field(..., description="Predicted category")
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Prediction confidence (0-1)"
    )
    top_3_predictions: List[PredictionDetail] = Field(
        ...,
        description="Top 3 predictions with confidences"
    )
    requires_review: bool = Field(
        ...,
        description="Whether prediction needs human review"
    )
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "REQ-2024-001234",
                "category": "Network",
                "confidence": 0.92,
                "top_3_predictions": [
                    {"category": "Network", "confidence": 0.92},
                    {"category": "Access Management", "confidence": 0.05},
                    {"category": "Software", "confidence": 0.02}
                ],
                "requires_review": False,
                "model_version": "v1.2.0",
                "latency_ms": 45.2,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class BatchTicketRequest(BaseModel):
    """Request schema for batch classification."""
    
    tickets: List[TicketRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of tickets to classify"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch classification."""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )
    total_latency_ms: float = Field(
        ...,
        description="Total processing time"
    )
    batch_size: int = Field(..., description="Number of tickets processed")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Current model version")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "v1.2.0",
                "uptime_seconds": 3600.5
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request ID if available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Subject must not be empty",
                "request_id": "REQ-2024-001234"
            }
        }
