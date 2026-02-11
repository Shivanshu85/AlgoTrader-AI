"""
Monitoring and Observability Utilities

Provides:
- Prometheus metrics collection
- Performance monitoring
- Request/response tracking
- Health metric exportation
"""

from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest,
    CollectorRegistry, REGISTRY
)
from functools import wraps
from typing import Callable, Any
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

class MetricsCollector:
    """Prometheus metrics for stock prediction API"""
    
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        """Initialize metrics collector"""
        self.registry = registry
        
        # Request metrics
        self.predictions_total = Counter(
            'predictions_total',
            'Total number of predictions served',
            ['model', 'status'],
            registry=registry
        )
        
        self.prediction_errors_total = Counter(
            'prediction_errors_total',
            'Total number of prediction errors',
            ['error_type'],
            registry=registry
        )
        
        self.http_requests_duration_seconds = Histogram(
            'http_requests_duration_seconds',
            'HTTP request latency in seconds',
            ['method', 'endpoint', 'status'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=registry
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total number of cache hits',
            registry=registry
        )
        
        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total number of cache misses',
            registry=registry
        )
        
        self.cache_size = Gauge(
            'cache_size',
            'Current size of prediction cache',
            registry=registry
        )
        
        # Model metrics
        self.model_inference_duration_seconds = Histogram(
            'model_inference_duration_seconds',
            'Model inference latency in seconds',
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            registry=registry
        )
        
        self.model_prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Model prediction confidence scores',
            buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
            registry=registry
        )
        
        self.model_version_timestamp_seconds = Gauge(
            'model_version_timestamp_seconds',
            'Timestamp of current model version',
            registry=registry
        )
        
        # Feature metrics
        self.features_validated_total = Counter(
            'features_validated_total',
            'Total features validated',
            ['status'],
            registry=registry
        )
        
        # Database metrics
        self.database_queries_total = Counter(
            'database_queries_total',
            'Total database queries',
            ['query_type', 'status'],
            registry=registry
        )
        
        self.database_query_duration_seconds = Histogram(
            'database_query_duration_seconds',
            'Database query latency in seconds',
            ['query_type'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=registry
        )
        
        # Health metrics
        self.api_uptime_seconds = Gauge(
            'api_uptime_seconds',
            'API uptime in seconds',
            registry=registry
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=registry
        )
    
    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry)


# Global metrics instance
metrics = MetricsCollector()


# ============================================================================
# DECORATORS FOR AUTOMATIC INSTRUMENTATION
# ============================================================================

def track_prediction(func: Callable) -> Callable:
    """Decorator to track prediction calls"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            metrics.model_inference_duration_seconds.observe(duration)
            
            # Track confidence if in result
            if isinstance(result, dict) and 'confidence' in result:
                metrics.model_prediction_confidence.observe(result['confidence'])
            
            metrics.predictions_total.labels(model='lstm-attention', status='success').inc()
            logger.info(f"Prediction successful in {duration:.4f}s")
            
            return result
        
        except Exception as e:
            metrics.predictions_total.labels(model='lstm-attention', status='error').inc()
            metrics.prediction_errors_total.labels(error_type=type(e).__name__).inc()
            logger.error(f"Prediction failed: {e}")
            raise
    
    return wrapper


def track_http_request(method: str = 'GET', endpoint: str = '/') -> Callable:
    """Decorator to track HTTP requests"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                metrics.http_requests_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status
                ).observe(duration)
        
        return wrapper
    return decorator


def track_cache_operation(operation: str) -> Callable:
    """Decorator to track cache operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)
            
            if operation == 'hit':
                metrics.cache_hits_total.inc()
            elif operation == 'miss':
                metrics.cache_misses_total.inc()
            
            return result
        
        return wrapper
    return decorator


def track_database_query(query_type: str) -> Callable:
    """Decorator to track database queries"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                metrics.database_query_duration_seconds.labels(query_type=query_type).observe(duration)
                metrics.database_queries_total.labels(query_type=query_type, status=status).inc()
        
        return wrapper
    return decorator


# ============================================================================
# HEALTH CHECK REPORTER
# ============================================================================

class HealthReporter:
    """Report system health metrics"""
    
    def __init__(self):
        """Initialize health reporter"""
        self.start_time = datetime.now()
    
    def update_uptime(self):
        """Update API uptime metric"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        metrics.api_uptime_seconds.set(uptime)
    
    def update_model_version(self, version: str):
        """Update model version metrics"""
        metrics.model_version_timestamp_seconds.set(datetime.now().timestamp())
    
    def update_cache_size(self, size: int):
        """Update cache size metric"""
        metrics.cache_size.set(size)
    
    def update_active_connections(self, count: int):
        """Update active connection count"""
        metrics.active_connections.set(count)
    
    def report_feature_validation(self, valid: bool):
        """Report feature validation result"""
        status = 'valid' if valid else 'invalid'
        metrics.features_validated_total.labels(status=status).inc()


# Global health reporter instance
health_reporter = HealthReporter()


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.request_times = []
        self.error_count = 0
        self.total_requests = 0
    
    def record_request(self, duration: float, success: bool = True):
        """Record a request"""
        self.request_times.append(duration)
        self.total_requests += 1
        
        if not success:
            self.error_count += 1
        
        # Keep only last 1000 requests
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        if not self.request_times:
            return {}
        
        times = sorted(self.request_times)
        
        return {
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.total_requests if self.total_requests > 0 else 0,
            'min_latency': min(times),
            'max_latency': max(times),
            'mean_latency': sum(times) / len(times),
            'median_latency': times[len(times) // 2],
            'p95_latency': times[int(len(times) * 0.95)],
            'p99_latency': times[int(len(times) * 0.99)],
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# ============================================================================
# METRICS UTILITIES
# ============================================================================

def export_metrics() -> str:
    """Export all metrics as Prometheus text format"""
    return metrics.get_metrics().decode('utf-8')


def get_health_metrics() -> dict:
    """Get current health metrics"""
    health_reporter.update_uptime()
    
    return {
        'uptime_seconds': (datetime.now() - health_reporter.start_time).total_seconds(),
        'predictions_total': metrics.predictions_total._value.get(),
        'errors_total': metrics.prediction_errors_total._value.get(),
        'cache_size': int(metrics.cache_size._value.get()),
        'active_connections': int(metrics.active_connections._value.get()),
    }


def reset_metrics():
    """Reset all metrics (for testing)"""
    # Note: Prometheus counters cannot be reset in production
    # This is mainly for testing purposes
    logger.warning("Resetting metrics (test mode only)")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create metrics
    collector = MetricsCollector()
    
    # Record some metrics
    collector.predictions_total.labels(model='lstm', status='success').inc(5)
    collector.cache_hits_total.inc(3)
    collector.cache_size.set(250)
    
    # Export metrics
    print("Metrics:")
    print(export_metrics())
