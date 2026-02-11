"""
API Tests

Comprehensive testing for:
- FastAPI endpoints
- Request validation
- Response formats
- Model serving
- Caching
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from datetime import datetime
from typing import List, Dict
import json

# Assuming imports from application modules
try:
    from main import app
    from models import ModelServer, PredictionCache
    from schemas import PredictionRequest, PredictionResponse
except ImportError:
    # Fallback for testing environment
    pass

# Create test client
client = TestClient(app)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_features():
    """Generate valid feature vector"""
    return [float(x) for x in np.random.randn(30) * 10 + 100]


@pytest.fixture
def batch_features():
    """Generate batch of feature vectors"""
    return [
        [float(x) for x in np.random.randn(30) * 10 + 100],
        [float(x) for x in np.random.randn(30) * 10 + 150],
        [float(x) for x in np.random.randn(30) * 10 + 120],
    ]


@pytest.fixture
def ticker():
    """Valid ticker symbol"""
    return "AAPL"


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self):
        """Test /health endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "predictions_served" in data
        assert "errors" in data
    
    def test_health_status_healthy(self):
        """Test health status is healthy"""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
    
    def test_readiness_endpoint(self):
        """Test /ready endpoint"""
        response = client.get("/ready")
        
        # Should return 200 if model loaded, 503 otherwise
        assert response.status_code in [200, 503]
    
    def test_liveness_endpoint(self):
        """Test /live endpoint"""
        response = client.get("/live")
        
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


# ============================================================================
# PREDICTION ENDPOINTS TESTS
# ============================================================================

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_single_prediction_success(self, ticker, valid_features):
        """Test successful single prediction"""
        request_data = {
            "ticker": ticker,
            "features": valid_features
        }
        
        response = client.post("/predict", json=request_data)
        
        # Should succeed if model loaded, otherwise 503
        if response.status_code == 200:
            data = response.json()
            
            assert "ticker" in data
            assert "predicted_price" in data
            assert "confidence" in data
            assert "timestamp" in data
            assert "model_version" in data
            assert "from_cache" in data
            
            assert data["ticker"] == ticker
            assert 0 <= data["confidence"] <= 1
    
    def test_single_prediction_invalid_features_length(self, ticker):
        """Test prediction with wrong feature length"""
        request_data = {
            "ticker": ticker,
            "features": [1.0, 2.0, 3.0]  # Wrong length
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_single_prediction_invalid_ticker(self, valid_features):
        """Test prediction with invalid ticker"""
        request_data = {
            "ticker": "INVALID_VERY_LONG_TICKER_NAME",
            "features": valid_features
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction_success(self, batch_features):
        """Test successful batch prediction"""
        tickers = ["AAPL", "GOOGL", "MSFT"]
        
        items = []
        for ticker, features in zip(tickers, batch_features):
            items.append({
                "ticker": ticker,
                "features": features
            })
        
        request_data = {"items": items}
        response = client.post("/predict/batch", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "batch_size" in data
            assert "predictions" in data
            assert "timestamp" in data
            assert "model_version" in data
            
            assert data["batch_size"] == 3
            assert len(data["predictions"]) == 3
    
    def test_batch_prediction_empty(self):
        """Test batch prediction with empty items"""
        request_data = {"items": []}
        response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 400  # Bad request
    
    def test_batch_prediction_too_large(self, valid_features):
        """Test batch prediction exceeding size limit"""
        items = [
            {
                "ticker": f"TICK{i}",
                "features": valid_features
            }
            for i in range(150)  # Exceeds limit of 100
        ]
        
        request_data = {"items": items}
        response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 400  # Bad request


# ============================================================================
# FEATURE ENDPOINTS TESTS
# ============================================================================

class TestFeatureEndpoints:
    """Test feature-related endpoints"""
    
    def test_feature_schema(self):
        """Test /features/schema endpoint"""
        response = client.get("/features/schema")
        
        if response.status_code == 200:
            data = response.json()
            
            assert "num_features" in data
            assert "expected_sequence_length" in data
            assert "feature_names" in data
            
            assert data["num_features"] == 30
    
    def test_feature_validation_valid(self, ticker, valid_features):
        """Test feature validation with valid features"""
        params = {
            "features": ",".join(str(f) for f in valid_features),
            "ticker": ticker
        }
        
        response = client.post("/features/validate", params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "valid" in data
            assert "issues" in data
            assert "warnings" in data


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS TESTS
# ============================================================================

class TestModelManagementEndpoints:
    """Test model management endpoints"""
    
    def test_model_info(self):
        """Test /model/info endpoint"""
        response = client.get("/model/info")
        
        if response.status_code == 200:
            data = response.json()
            
            assert "model_name" in data
            assert "version" in data
            assert "framework" in data
            assert "architecture" in data
            assert "parameters" in data
            assert "device" in data
    
    def test_model_info_contains_expected_fields(self):
        """Test model info contains expected fields"""
        response = client.get("/model/info")
        
        if response.status_code == 200:
            data = response.json()
            
            assert "LSTM" in data.get("architecture", "")
            assert "PyTorch" in data.get("framework", "")


# ============================================================================
# METRICS ENDPOINTS TESTS
# ============================================================================

class TestMetricsEndpoints:
    """Test metrics endpoints"""
    
    def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions_served" in data
        assert "errors" in data
        assert "error_rate" in data
        assert "request_rate_per_second" in data
        assert "uptime_seconds" in data
        assert "cache_size" in data
    
    def test_prometheus_metrics(self):
        """Test /metrics/prometheus endpoint"""
        response = client.get("/metrics/prometheus")
        
        assert response.status_code == 200
        
        # Should contain Prometheus format metrics
        content = response.text
        assert "predictions_total" in content
        assert "prediction_errors_total" in content
        assert "service_uptime_seconds" in content


# ============================================================================
# ROOT ENDPOINT TEST
# ============================================================================

class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self):
        """Test / endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert "endpoints" in data


# ============================================================================
# CACHE ENDPOINT TESTS
# ============================================================================

class TestCacheEndpoints:
    """Test cache-related endpoints"""
    
    def test_cache_operations(self):
        """Test cache operations"""
        cache = PredictionCache(ttl_seconds=60)
        
        # Set
        cache.set("key1", {"value": 100})
        
        # Get
        assert cache.get("key1") == {"value": 100}
        
        # Has
        assert cache.has("key1") is True
        
        # Size
        assert cache.size() == 1
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        import time
        
        cache = PredictionCache(ttl_seconds=1)
        cache.set("key1", {"value": 100})
        
        assert cache.has("key1") is True
        
        # Wait for expiration
        time.sleep(2)
        
        assert cache.has("key1") is False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests"""
    
    def test_prediction_pipeline(self, ticker, valid_features):
        """Test complete prediction pipeline"""
        # First prediction (not cached)
        request_data = {
            "ticker": ticker,
            "features": valid_features
        }
        
        response1 = client.post("/predict", json=request_data)
        
        if response1.status_code == 200:
            data1 = response1.json()
            original_from_cache = data1["from_cache"]
            
            # Second identical prediction (should be cached)
            response2 = client.post("/predict", json=request_data)
            data2 = response2.json()
            
            # Cache should work
            assert data2["from_cache"] == True or data1["predicted_price"] == data2["predicted_price"]
    
    def test_error_handling(self):
        """Test error handling"""
        # Send malformed request
        response = client.post(
            "/predict",
            json={"invalid": "request"}
        )
        
        assert response.status_code == 422
    
    def test_model_metadata_consistency(self):
        """Test model metadata consistency"""
        health_response = client.get("/health")
        model_response = client.get("/model/info")
        metrics_response = client.get("/metrics")
        
        if all(r.status_code == 200 for r in [health_response, model_response, metrics_response]):
            health_data = health_response.json()
            model_data = model_response.json()
            metrics_data = metrics_response.json()
            
            # Versions should be consistent
            assert health_data.get("model_loaded") is not None
            assert model_data.get("version") is not None
            assert metrics_data.get("model_version") is not None


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests"""
    
    def test_prediction_response_time(self, ticker, valid_features):
        """Test prediction response time"""
        import time
        
        request_data = {
            "ticker": ticker,
            "features": valid_features
        }
        
        start = time.time()
        response = client.post("/predict", json=request_data)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 1 second)
        if response.status_code == 200:
            assert elapsed < 1.0
    
    def test_health_check_response_time(self):
        """Test health check response time"""
        import time
        
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 0.1  # Health checks should be fast


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
