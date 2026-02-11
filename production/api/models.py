"""
Model Serving Module

Handles:
- Model loading from MLflow
- Inference execution
- Prediction caching
- Feature validation
"""

import torch
import torch.nn as nn
import numpy as np
import mlflow
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict
import json

logger = logging.getLogger(__name__)


class PredictionCache:
    """LRU cache for predictions"""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000):
        """
        Initialize cache
        
        Args:
            ttl_seconds: Time to live in seconds
            max_size: Maximum cache size
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, datetime] = {}
    
    def set(self, key: str, value: Any):
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove expired items
        self._cleanup_expired()
        
        # Enforce max size
        if len(self.cache) >= self.max_size:
            # Remove oldest item (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cache value
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.has(key):
            return None
        
        # Move to end (mark as recently used in LRU)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def has(self, key: str) -> bool:
        """
        Check if key exists and not expired
        
        Args:
            key: Cache key
            
        Returns:
            True if valid entry exists
        """
        if key not in self.cache:
            return False
        
        # Check expiration
        age = (datetime.now() - self.timestamps[key]).total_seconds()
        if age > self.ttl_seconds:
            del self.cache[key]
            del self.timestamps[key]
            return False
        
        return True
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            k for k, ts in self.timestamps.items()
            if (now - ts).total_seconds() > self.ttl_seconds
        ]
        for k in expired_keys:
            del self.cache[k]
            del self.timestamps[k]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get cache size"""
        self._cleanup_expired()
        return len(self.cache)


class ModelServer:
    """Model serving with MLflow integration"""
    
    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5000",
        model_name: str = "stock-price-lstm",
        stage: str = "Production"
    ):
        """
        Initialize model server
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            model_name: Model name in MLflow
            stage: Model stage (Production, Staging, etc.)
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.model_name = model_name
        self.stage = stage
        
        self.model: Optional[nn.Module] = None
        self.model_version: str = "unknown"
        self.device: torch.device = self._get_device()
        self.input_dim: int = 30
        self.feature_scaler: Optional[Any] = None
        self.normalizer: Optional[Any] = None
    
    def _get_device(self) -> torch.device:
        """Get appropriate device"""
        if torch.cuda.is_available():
            logger.info("Using GPU")
            return torch.device('cuda')
        else:
            logger.info("Using CPU")
            return torch.device('cpu')
    
    def load_model(self):
        """Load model from MLflow"""
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            logger.info(f"Loading model: {self.model_name} (Stage: {self.stage})")
            
            # Load model from MLflow Model Registry
            try:
                model_uri = f"models:/{self.model_name}/{self.stage}"
                self.model = mlflow.pytorch.load_model(model_uri)
                self.model_version = self.stage
                logger.info(f"✅ Model loaded from registry: {model_uri}")
            except:
                # Fallback: load from runs
                logger.warning("Could not load from registry, trying runs...")
                latest_run = self._get_latest_run()
                if latest_run:
                    model_uri = f"runs:/{latest_run['run_id']}/model"
                    self.model = mlflow.pytorch.load_model(model_uri)
                    self.model_version = latest_run.get('tags.mlflow.runName', 'latest')
                    logger.info(f"✅ Model loaded from run: {model_uri}")
                else:
                    raise Exception("No models found in MLflow")
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model moved to device: {self.device}")
            logger.info(f"Model parameters: {self.get_parameter_count():,}")
        
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            # Fallback to mock model for demonstration
            logger.info("Using mock model for demonstration")
            self._create_mock_model()
    
    def _get_latest_run(self) -> Optional[Dict]:
        """Get latest run from MLflow"""
        try:
            runs = mlflow.search_runs(
                max_results=1,
                order_by=["start_time DESC"]
            )
            if len(runs) > 0:
                return {
                    'run_id': runs.iloc[0]['run_id'],
                    'tags.mlflow.runName': runs.iloc[0].get('tags.mlflow.runName', 'latest')
                }
        except:
            pass
        return None
    
    def _create_mock_model(self):
        """Create mock model for demonstration"""
        # Simple model for testing
        self.model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)
        self.model.eval()
    
    def predict(
        self,
        features: np.ndarray,
        ticker: str = "UNKNOWN"
    ) -> Dict[str, float]:
        """
        Make prediction
        
        Args:
            features: Input features (30 values)
            ticker: Stock ticker
            
        Returns:
            Prediction dictionary with price and confidence
        """
        try:
            # Validate input
            features = self._validate_and_process_features(features)
            
            # Convert to tensor
            features_tensor = torch.from_numpy(features).float().to(self.device)
            
            # Add batch dimension if needed
            if features_tensor.dim() == 1:
                features_tensor = features_tensor.unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                prediction = self.model(features_tensor)
            
            # Extract value
            pred_value = float(prediction.squeeze().cpu().numpy())
            
            # Calculate confidence (higher std in training data = lower confidence)
            confidence = self._estimate_confidence(features, pred_value, ticker)
            
            logger.debug(f"Prediction for {ticker}: {pred_value:.4f} (confidence: {confidence:.2%})")
            
            return {
                'prediction': pred_value,
                'confidence': confidence,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            raise
    
    def _validate_and_process_features(self, features: np.ndarray) -> np.ndarray:
        """
        Validate and process features
        
        Args:
            features: Input features
            
        Returns:
            Processed features
        """
        # Ensure numpy array
        if isinstance(features, list):
            features = np.array(features)
        
        # Check shape
        if features.ndim == 1:
            if len(features) != 30:
                raise ValueError(f"Expected 30 features, got {len(features)}")
        elif features.ndim == 2:
            if features.shape[1] != 30:
                raise ValueError(f"Expected 30 features per sample, got {features.shape[1]}")
        else:
            raise ValueError(f"Expected 1D or 2D array, got {features.ndim}D")
        
        # Check for NaN/Inf
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError("Features contain NaN or Inf values")
        
        # Normalize to reasonable range if needed
        features = np.clip(features, -1e5, 1e5)
        
        return features
    
    def validate_features(
        self,
        features: np.ndarray,
        ticker: str
    ) -> Dict[str, Any]:
        """
        Validate features
        
        Args:
            features: Input features
            ticker: Stock ticker
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check shape
        if features.ndim != 1 or len(features) != 30:
            issues.append(f"Expected 30 features, got {len(features)}")
        
        # Check for NaN/Inf
        if np.isnan(features).any():
            issues.append("Features contain NaN values")
        if np.isinf(features).any():
            issues.append("Features contain Inf values")
        
        # Check ranges
        if np.any(np.abs(features) > 1e5):
            warnings.append("Some features are very large (>1e5)")
        
        # Check histogram
        if np.std(features) < 0.01:
            warnings.append("Features have very low variance")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'statistics': {
                'mean': float(np.mean(features)),
                'std': float(np.std(features)),
                'min': float(np.min(features)),
                'max': float(np.max(features)),
            }
        }
    
    def _estimate_confidence(
        self,
        features: np.ndarray,
        prediction: float,
        ticker: str
    ) -> float:
        """
        Estimate prediction confidence
        
        Args:
            features: Input features
            prediction: Predicted value
            ticker: Stock ticker
            
        Returns:
            Confidence score 0-1
        """
        # Simple confidence heuristic
        # Based on feature variance and prediction stability
        
        base_confidence = 0.85
        
        # Reduce confidence if features have low variance
        feature_std = float(np.std(features))
        if feature_std < 0.1:
            base_confidence -= 0.10
        
        # Reduce confidence for extreme predictions
        if abs(prediction) > 10000:
            base_confidence -= 0.05
        
        # Ensure in valid range
        return np.clip(base_confidence, 0.5, 0.95)
    
    def get_input_dimension(self) -> int:
        """Get input dimension"""
        return self.input_dim
    
    def get_parameter_count(self) -> int:
        """Get total parameter count"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model server cleanup complete")


class ModelVersionManager:
    """Manage model versions"""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize manager"""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_metadata(
        self,
        version: str,
        metrics: Dict[str, float],
        hyperparams: Dict[str, Any]
    ):
        """
        Save model metadata
        
        Args:
            version: Model version
            metrics: Performance metrics
            hyperparams: Hyperparameters
        """
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'hyperparams': hyperparams,
        }
        
        metadata_path = self.models_dir / f"metadata_{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata for version {version}")
    
    def get_model_metadata(self, version: str) -> Optional[Dict]:
        """Get model metadata"""
        metadata_path = self.models_dir / f"metadata_{version}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None


if __name__ == "__main__":
    # Example usage
    logger.basicConfig(level=logging.INFO)
    
    # Test cache
    cache = PredictionCache(ttl_seconds=60, max_size=100)
    cache.set("key1", {"value": 100})
    print("Cache test:", cache.get("key1"))
    
    # Test model server
    server = ModelServer()
    print("Parameter count:", server.get_parameter_count())
