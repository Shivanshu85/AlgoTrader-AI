"""
FastAPI Application for Stock Price Prediction Service

Provides:
- REST API endpoints for predictions
- Model serving from MLflow
- Health checks and metrics
- Batch prediction support
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import torch
import mlflow
import json
from pathlib import Path

# Import application modules
from schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, MetricsResponse
)
from models import ModelServer, PredictionCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="Production API for LSTM-based stock price prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZIP compression
app.add_middleware(GZIPMiddleware, minimum_size=1000)

# Global state
model_server: Optional[ModelServer] = None
prediction_cache: Optional[PredictionCache] = None
start_time: datetime = datetime.now()
prediction_count: int = 0
error_count: int = 0


# ============================================================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model and cache on startup"""
    global model_server, prediction_cache, start_time
    
    start_time = datetime.now()
    
    try:
        logger.info("Initializing model server...")
        model_server = ModelServer()
        
        logger.info("Loading model from MLflow...")
        model_server.load_model()
        
        logger.info("Initializing prediction cache...")
        prediction_cache = PredictionCache(ttl_seconds=3600)
        
        logger.info("✅ Application startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model_server, prediction_cache
    
    logger.info("Shutting down application...")
    
    if model_server:
        model_server.cleanup()
        logger.info("Model server stopped")
    
    if prediction_cache:
        prediction_cache.clear()
        logger.info("Cache cleared")
    
    logger.info("✅ Application shutdown complete")


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    
    Returns:
        HealthResponse with status and uptime
    """
    uptime = datetime.now() - start_time
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime.total_seconds(),
        timestamp=datetime.now(),
        model_loaded=model_server is not None,
        predictions_served=prediction_count,
        errors=error_count,
    )


@app.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check for load balancers
    
    Returns:
        Status response
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"status": "ready"}


@app.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check for load balancers
    
    Returns:
        Status response
    """
    return {"status": "alive"}


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Single prediction endpoint
    
    Args:
        request: PredictionRequest with features
        
    Returns:
        PredictionResponse with prediction
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    global prediction_count, error_count
    
    try:
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"Prediction request for ticker: {request.ticker}")
        
        # Check cache
        cache_key = request.get_cache_key()
        if prediction_cache and prediction_cache.has(cache_key):
            logger.info("Cache hit!")
            cached = prediction_cache.get(cache_key)
            cached['from_cache'] = True
            prediction_count += 1
            return PredictionResponse(**cached)
        
        # Perform prediction
        prediction_data = model_server.predict(
            features=np.array(request.features),
            ticker=request.ticker
        )
        
        response = PredictionResponse(
            ticker=request.ticker,
            predicted_price=float(prediction_data['prediction']),
            confidence=float(prediction_data.get('confidence', 0.85)),
            timestamp=datetime.now(),
            model_version=model_server.model_version,
            from_cache=False,
        )
        
        # Cache result
        if prediction_cache:
            prediction_cache.set(cache_key, response.dict())
        
        prediction_count += 1
        logger.info(f"✅ Prediction complete: {response.predicted_price:.2f}")
        
        return response
    
    except Exception as e:
        error_count += 1
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch prediction endpoint
    
    Args:
        request: BatchPredictionRequest with multiple feature sets
        
    Returns:
        BatchPredictionResponse with predictions
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    global prediction_count, error_count
    
    try:
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if len(request.items) == 0:
            raise HTTPException(status_code=400, detail="Empty batch")
        
        if len(request.items) > 100:
            raise HTTPException(status_code=400, detail="Batch size exceeds 100")
        
        logger.info(f"Batch prediction request: {len(request.items)} items")
        
        predictions = []
        for item in request.items:
            try:
                prediction_data = model_server.predict(
                    features=np.array(item.features),
                    ticker=item.ticker
                )
                
                predictions.append({
                    'ticker': item.ticker,
                    'predicted_price': float(prediction_data['prediction']),
                    'confidence': float(prediction_data.get('confidence', 0.85)),
                })
            except Exception as e:
                logger.error(f"Error predicting for {item.ticker}: {e}")
                error_count += 1
                predictions.append({
                    'ticker': item.ticker,
                    'error': str(e),
                })
        
        response = BatchPredictionResponse(
            batch_size=len(request.items),
            predictions=predictions,
            timestamp=datetime.now(),
            model_version=model_server.model_version,
        )
        
        prediction_count += len(request.items)
        logger.info(f"✅ Batch prediction complete: {len(predictions)} predictions")
        
        return response
    
    except Exception as e:
        error_count += 1
        logger.error(f"❌ Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FEATURE ENDPOINTS
# ============================================================================

@app.post("/features/validate")
async def validate_features(
    features: List[float] = Query(..., description="Feature values"),
    ticker: str = Query(..., description="Stock ticker")
) -> Dict[str, Any]:
    """
    Validate feature values
    
    Args:
        features: Feature values
        ticker: Stock ticker
        
    Returns:
        Validation results
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        validation = model_server.validate_features(
            features=np.array(features),
            ticker=ticker
        )
        return validation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/features/schema")
async def get_feature_schema() -> Dict[str, Any]:
    """
    Get feature schema
    
    Returns:
        Feature schema with dimensions and names
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "num_features": model_server.get_input_dimension(),
        "expected_sequence_length": 30,
        "feature_names": [
            "close_price", "volume", "rsi", "macd", "sma_20",
            "sma_50", "ema_12", "ema_26", "bollinger_upper", "bollinger_lower"
        ]
    }


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/model/info")
async def get_model_info() -> Dict[str, Any]:
    """
    Get model information
    
    Returns:
        Model metadata and version
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "LSTM-Attention-StockPrice",
        "version": model_server.model_version,
        "framework": "PyTorch",
        "input_size": model_server.get_input_dimension(),
        "architecture": "LSTM with Multi-Head Attention",
        "parameters": model_server.get_parameter_count(),
        "device": str(model_server.device),
        "loaded_at": start_time.isoformat(),
    }


@app.post("/model/reload")
async def reload_model() -> Dict[str, str]:
    """
    Reload model from MLflow
    
    Returns:
        Status message
    """
    global model_server
    
    try:
        logger.info("Reloading model...")
        if model_server:
            model_server.cleanup()
        
        model_server = ModelServer()
        model_server.load_model()
        
        logger.info("✅ Model reloaded successfully")
        return {"status": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"❌ Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# METRICS AND MONITORING ENDPOINTS
# ============================================================================

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """
    Get service metrics
    
    Returns:
        MetricsResponse with performance data
    """
    uptime = datetime.now() - start_time
    uptime_seconds = uptime.total_seconds()
    
    # Calculate request rate
    request_rate = prediction_count / uptime_seconds if uptime_seconds > 0 else 0
    error_rate = error_count / prediction_count if prediction_count > 0 else 0
    
    return MetricsResponse(
        predictions_served=prediction_count,
        errors=error_count,
        error_rate=error_rate,
        request_rate_per_second=request_rate,
        uptime_seconds=uptime_seconds,
        cache_size=prediction_cache.size() if prediction_cache else 0,
        model_version=model_server.model_version if model_server else "unknown",
    )


@app.get("/metrics/prometheus")
async def get_prometheus_metrics() -> str:
    """
    Get metrics in Prometheus format
    
    Returns:
        Prometheus-formatted metrics
    """
    uptime = datetime.now() - start_time
    uptime_seconds = uptime.total_seconds()
    
    metrics = f"""# HELP predictions_total Total predictions served
# TYPE predictions_total counter
predictions_total {prediction_count}

# HELP prediction_errors_total Total prediction errors
# TYPE prediction_errors_total counter
prediction_errors_total {error_count}

# HELP service_uptime_seconds Service uptime
# TYPE service_uptime_seconds gauge
service_uptime_seconds {uptime_seconds}

# HELP cache_size Current cache size
# TYPE cache_size gauge
cache_size {prediction_cache.size() if prediction_cache else 0}
"""
    return metrics


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint with API information
    
    Returns:
        API metadata
    """
    return {
        "name": "Stock Price Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "metrics": "/metrics",
            "docs": "/docs",
            "openapi": "/openapi.json",
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

@app.post("/cache/clear")
async def clear_cache(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Clear prediction cache
    
    Args:
        background_tasks: Background task queue
        
    Returns:
        Status message
    """
    def clear_cache_task():
        if prediction_cache:
            prediction_cache.clear()
            logger.info("Cache cleared")
    
    background_tasks.add_task(clear_cache_task)
    return {"status": "Cache clear scheduled"}


if __name__ == "__main__":
    import uvicorn
    
    # Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
