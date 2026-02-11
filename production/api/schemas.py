"""
Request and Response Schema Definitions

Defines Pydantic models for API validation and documentation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


class PredictionRequest(BaseModel):
    """Single prediction request"""
    
    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    features: List[float] = Field(..., description="Feature vector (30 values)", min_items=30, max_items=30)
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL",
                "features": [150.0] * 30  # Example 30 features
            }
        }
    
    @validator('ticker')
    def validate_ticker(cls, v):
        """Validate ticker format"""
        if not v or len(v) > 10:
            raise ValueError('Invalid ticker format')
        return v.upper()
    
    @validator('features')
    def validate_features(cls, v):
        """Validate feature values"""
        if len(v) != 30:
            raise ValueError('Must provide exactly 30 features')
        if not all(isinstance(f, (int, float)) for f in v):
            raise ValueError('All features must be numeric')
        if any(not (-1e6 < f < 1e6) for f in v):
            raise ValueError('Feature values out of acceptable range')
        return v
    
    def get_cache_key(self) -> str:
        """Generate cache key"""
        features_hash = hash(tuple([round(f, 4) for f in self.features]))
        return f"{self.ticker}_{features_hash}"


class PredictionResponse(BaseModel):
    """Single prediction response"""
    
    ticker: str = Field(..., description="Stock ticker")
    predicted_price: float = Field(..., description="Predicted price")
    confidence: float = Field(..., description="Prediction confidence (0-1)", ge=0, le=1)
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version")
    from_cache: bool = Field(False, description="Whether prediction was cached")
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL",
                "predicted_price": 175.32,
                "confidence": 0.87,
                "timestamp": "2024-02-12T10:30:00",
                "model_version": "v1.0.0",
                "from_cache": False
            }
        }


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction request"""
    
    ticker: str = Field(..., description="Stock ticker")
    features: List[float] = Field(..., description="Feature vector", min_items=30, max_items=30)


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    
    items: List[BatchPredictionItem] = Field(..., description="List of prediction requests", min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "ticker": "AAPL",
                        "features": [150.0] * 30
                    },
                    {
                        "ticker": "GOOGL",
                        "features": [2800.0] * 30
                    }
                ]
            }
        }


class BatchPredictionItemResponse(BaseModel):
    """Single item in batch prediction response"""
    
    ticker: str
    predicted_price: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    
    batch_size: int = Field(..., description="Number of items in batch")
    predictions: List[BatchPredictionItemResponse] = Field(..., description="Predictions")
    timestamp: datetime = Field(..., description="Response timestamp")
    model_version: str = Field(..., description="Model version")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_size": 2,
                "predictions": [
                    {
                        "ticker": "AAPL",
                        "predicted_price": 175.32,
                        "confidence": 0.87
                    },
                    {
                        "ticker": "GOOGL",
                        "predicted_price": 2850.50,
                        "confidence": 0.82
                    }
                ],
                "timestamp": "2024-02-12T10:30:00",
                "model_version": "v1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    timestamp: datetime = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    predictions_served: int = Field(..., description="Total predictions served")
    errors: int = Field(..., description="Total errors encountered")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "uptime_seconds": 3600.5,
                "timestamp": "2024-02-12T10:30:00",
                "model_loaded": True,
                "predictions_served": 150,
                "errors": 2
            }
        }


class MetricsResponse(BaseModel):
    """Metrics response"""
    
    predictions_served: int = Field(..., description="Total predictions")
    errors: int = Field(..., description="Total errors")
    error_rate: float = Field(..., description="Error rate 0-1", ge=0, le=1)
    request_rate_per_second: float = Field(..., description="Requests per second")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    cache_size: int = Field(..., description="Current cache size")
    model_version: str = Field(..., description="Model version")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions_served": 1250,
                "errors": 5,
                "error_rate": 0.004,
                "request_rate_per_second": 0.35,
                "uptime_seconds": 3600.0,
                "cache_size": 420,
                "model_version": "v1.0.0"
            }
        }


class FeatureValidationResponse(BaseModel):
    """Feature validation response"""
    
    valid: bool = Field(..., description="Whether features are valid")
    issues: List[str] = Field(default=[], description="Validation issues if any")
    warnings: List[str] = Field(default=[], description="Warnings about features")
    statistics: Optional[Dict[str, float]] = Field(None, description="Feature statistics")


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    detail: Optional[str] = Field(None, description="Additional details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Model not loaded",
                "timestamp": "2024-02-12T10:30:00",
                "detail": "Model failed to load during initialization"
            }
        }


# Request/Response Examples for documentation

PREDICTION_REQUEST_EXAMPLE = {
    "ticker": "AAPL",
    "features": [
        150.5, 152.3, 151.8, 149.2, 150.1,  # Price series
        2.5, 3.1, 2.8, 2.9, 3.0,             # RSI values
        -0.5, -0.2, 0.1, 0.3, -0.1,          # MACD values
        148.0, 149.5, 150.2, 151.0, 150.8,   # SMA values
        0.8, 0.7, 0.9, 0.85, 0.75,           # Additional features
        2500000, 2600000, 2550000, 2700000, 2650000  # Volume
    ]
}

PREDICTION_RESPONSE_EXAMPLE = {
    "ticker": "AAPL",
    "predicted_price": 175.32,
    "confidence": 0.87,
    "timestamp": "2024-02-12T10:30:45.123456",
    "model_version": "v1.0.0",
    "from_cache": False
}

BATCH_PREDICTION_REQUEST_EXAMPLE = {
    "items": [
        {
            "ticker": "AAPL",
            "features": [150.5] * 30
        },
        {
            "ticker": "GOOGL",
            "features": [2850.0] * 30
        },
        {
            "ticker": "MSFT",
            "features": [380.0] * 30
        }
    ]
}

BATCH_PREDICTION_RESPONSE_EXAMPLE = {
    "batch_size": 3,
    "predictions": [
        {
            "ticker": "AAPL",
            "predicted_price": 175.32,
            "confidence": 0.87
        },
        {
            "ticker": "GOOGL",
            "predicted_price": 2850.50,
            "confidence": 0.82
        },
        {
            "ticker": "MSFT",
            "predicted_price": 385.75,
            "confidence": 0.84
        }
    ],
    "timestamp": "2024-02-12T10:30:45.123456",
    "model_version": "v1.0.0"
}
