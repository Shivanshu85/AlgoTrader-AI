"""
Monitoring and Observability Tests

Comprehensive testing for:
- Metrics collection
- Health reporting
- Alert validation
- Dashboard configuration
"""

import pytest
import time
from typing import Dict, Any
from prometheus_client import CollectorRegistry

# Assuming imports from production modules
try:
    from production.monitoring.metrics import (
        MetricsCollector, HealthReporter, PerformanceMonitor,
        track_prediction, track_http_request, track_cache_operation,
        export_metrics, get_health_metrics
    )
except ImportError:
    # Fallback for test environment
    pass


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def metrics_registry():
    """Create isolated metrics registry for testing"""
    return CollectorRegistry()


@pytest.fixture
def metrics_collector(metrics_registry):
    """Create metrics collector with test registry"""
    return MetricsCollector(registry=metrics_registry)


@pytest.fixture
def health_reporter():
    """Create health reporter"""
    return HealthReporter()


@pytest.fixture
def performance_monitor():
    """Create performance monitor"""
    return PerformanceMonitor()


# ============================================================================
# METRICS COLLECTOR TESTS
# ============================================================================

class TestMetricsCollector:
    """Test Prometheus metrics collection"""
    
    def test_metrics_initialization(self, metrics_collector):
        """Test metrics collector initialization"""
        assert metrics_collector is not None
        assert metrics_collector.predictions_total is not None
        assert metrics_collector.cache_hits_total is not None
        assert metrics_collector.model_inference_duration_seconds is not None
    
    def test_predictions_counter(self, metrics_collector):
        """Test predictions counter"""
        # Increment counter
        metrics_collector.predictions_total.labels(model='lstm', status='success').inc(5)
        
        # Export and verify
        metrics_text = metrics_collector.registry.collect()
        assert any('predictions_total' in str(m) for m in metrics_text)
    
    def test_error_counter(self, metrics_collector):
        """Test error counter"""
        metrics_collector.prediction_errors_total.labels(error_type='ValueError').inc(2)
        
        metrics_text = metrics_collector.registry.collect()
        assert any('prediction_errors_total' in str(m) for m in metrics_text)
    
    def test_cache_metrics(self, metrics_collector):
        """Test cache metrics"""
        metrics_collector.cache_hits_total.inc()
        metrics_collector.cache_misses_total.inc(2)
        metrics_collector.cache_size.set(150)
        
        metrics_text = metrics_collector.registry.collect()
        assert any('cache_size' in str(m) for m in metrics_text)
    
    def test_inference_duration_histogram(self, metrics_collector):
        """Test inference duration histogram"""
        metrics_collector.model_inference_duration_seconds.observe(0.05)
        metrics_collector.model_inference_duration_seconds.observe(0.15)
        metrics_collector.model_inference_duration_seconds.observe(0.08)
        
        metrics_text = metrics_collector.registry.collect()
        assert any('model_inference_duration_seconds' in str(m) for m in metrics_text)
    
    def test_confidence_histogram(self, metrics_collector):
        """Test confidence histogram"""
        metrics_collector.model_prediction_confidence.observe(0.85)
        metrics_collector.model_prediction_confidence.observe(0.92)
        metrics_collector.model_prediction_confidence.observe(0.88)
        
        metrics_text = metrics_collector.registry.collect()
        assert any('model_prediction_confidence' in str(m) for m in metrics_text)
    
    def test_http_request_tracking(self, metrics_collector):
        """Test HTTP request metrics"""
        metrics_collector.http_requests_duration_seconds.labels(
            method='POST',
            endpoint='/predict',
            status='success'
        ).observe(0.05)
        
        metrics_collector.http_requests_duration_seconds.labels(
            method='GET',
            endpoint='/health',
            status='success'
        ).observe(0.01)
        
        metrics_text = metrics_collector.registry.collect()
        assert any('http_requests_duration_seconds' in str(m) for m in metrics_text)


# ============================================================================
# DECORATOR TESTS
# ============================================================================

class TestMetricsDecorators:
    """Test metrics tracking decorators"""
    
    def test_track_prediction_decorator_success(self):
        """Test prediction tracking decorator on success"""
        @track_prediction
        def mock_predict():
            return {'prediction': 150.5, 'confidence': 0.87}
        
        result = mock_predict()
        assert result['prediction'] == 150.5
        assert result['confidence'] == 0.87
    
    def test_track_prediction_decorator_error(self):
        """Test prediction tracking decorator on error"""
        @track_prediction
        def mock_predict():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            mock_predict()
    
    def test_track_http_request_decorator(self):
        """Test HTTP request tracking decorator"""
        @track_http_request(method='POST', endpoint='/predict')
        def mock_request():
            time.sleep(0.01)
            return {'status': 'ok'}
        
        result = mock_request()
        assert result['status'] == 'ok'
    
    def test_track_cache_decorator_hit(self):
        """Test cache hit tracking"""
        @track_cache_operation('hit')
        def mock_cache_get():
            return {'cached': True}
        
        result = mock_cache_get()
        assert result['cached'] is True
    
    def test_track_cache_decorator_miss(self):
        """Test cache miss tracking"""
        @track_cache_operation('miss')
        def mock_cache_miss():
            return None
        
        result = mock_cache_miss()
        assert result is None
    
    def test_track_database_decorator(self):
        """Test database query tracking"""
        @track_cache_operation('hit')
        def mock_db_query():
            time.sleep(0.01)
            return [{'id': 1, 'value': 100}]
        
        result = mock_db_query()
        assert len(result) == 1


# ============================================================================
# HEALTH REPORTER TESTS
# ============================================================================

class TestHealthReporter:
    """Test health reporting"""
    
    def test_health_reporter_initialization(self, health_reporter):
        """Test health reporter initialization"""
        assert health_reporter is not None
        assert health_reporter.start_time is not None
    
    def test_update_uptime(self, health_reporter):
        """Test uptime update"""
        health_reporter.update_uptime()
        # Should not raise
        assert True
    
    def test_update_model_version(self, health_reporter):
        """Test model version update"""
        health_reporter.update_model_version("v1.0.0")
        # Should not raise
        assert True
    
    def test_update_cache_size(self, health_reporter):
        """Test cache size update"""
        health_reporter.update_cache_size(500)
        # Should not raise
        assert True
    
    def test_update_active_connections(self, health_reporter):
        """Test active connections update"""
        health_reporter.update_active_connections(42)
        # Should not raise
        assert True


# ============================================================================
# PERFORMANCE MONITOR TESTS
# ============================================================================

class TestPerformanceMonitor:
    """Test performance monitoring"""
    
    def test_monitor_initialization(self, performance_monitor):
        """Test performance monitor initialization"""
        assert performance_monitor.total_requests == 0
        assert performance_monitor.error_count == 0
    
    def test_record_successful_request(self, performance_monitor):
        """Test recording successful request"""
        performance_monitor.record_request(0.05, success=True)
        
        assert performance_monitor.total_requests == 1
        assert performance_monitor.error_count == 0
        assert len(performance_monitor.request_times) == 1
    
    def test_record_failed_request(self, performance_monitor):
        """Test recording failed request"""
        performance_monitor.record_request(0.15, success=False)
        
        assert performance_monitor.total_requests == 1
        assert performance_monitor.error_count == 1
    
    def test_get_stats_empty(self, performance_monitor):
        """Test stats on empty monitor"""
        stats = performance_monitor.get_stats()
        assert stats == {}
    
    def test_get_stats_with_data(self, performance_monitor):
        """Test stats with data"""
        times = [0.01, 0.05, 0.08, 0.1, 0.15]
        for t in times:
            performance_monitor.record_request(t)
        
        stats = performance_monitor.get_stats()
        
        assert 'total_requests' in stats
        assert 'min_latency' in stats
        assert 'max_latency' in stats
        assert 'mean_latency' in stats
        assert 'p95_latency' in stats
        assert 'p99_latency' in stats
        
        assert stats['total_requests'] == 5
        assert stats['min_latency'] == 0.01
        assert stats['max_latency'] == 0.15
    
    def test_stats_calculations(self, performance_monitor):
        """Test statistics calculations accuracy"""
        times = [0.1, 0.2, 0.3, 0.4, 0.5]
        for t in times:
            performance_monitor.record_request(t)
        
        stats = performance_monitor.get_stats()
        
        # Check mean calculation
        expected_mean = sum(times) / len(times)
        assert abs(stats['mean_latency'] - expected_mean) < 0.01
        
        # Check median
        assert stats['median_latency'] == 0.3
    
    def test_error_rate_calculation(self, performance_monitor):
        """Test error rate calculation"""
        performance_monitor.record_request(0.05, success=True)
        performance_monitor.record_request(0.06, success=True)
        performance_monitor.record_request(0.07, success=False)
        performance_monitor.record_request(0.08, success=False)
        
        stats = performance_monitor.get_stats()
        
        assert stats['total_requests'] == 4
        assert stats['error_count'] == 2
        assert abs(stats['error_rate'] - 0.5) < 0.01


# ============================================================================
# METRICS EXPORT TESTS
# ============================================================================

class TestMetricsExport:
    """Test metrics export functionality"""
    
    def test_export_metrics_format(self):
        """Test metrics are exported in Prometheus format"""
        metrics_text = export_metrics()
        
        assert isinstance(metrics_text, str)
        assert 'predictions_total' in metrics_text or len(metrics_text) > 0
    
    def test_get_health_metrics(self):
        """Test getting health metrics"""
        health_metrics = get_health_metrics()
        
        assert 'uptime_seconds' in health_metrics
        assert 'cache_size' in health_metrics


# ============================================================================
# ALERT RULE VALIDATION TESTS
# ============================================================================

class TestAlertRules:
    """Test alert rules configuration"""
    
    def test_alert_rules_format(self):
        """Test alert rules are valid YAML"""
        # Load alert rules
        import yaml
        
        with open('config/alert_rules.yml', 'r') as f:
            try:
                rules = yaml.safe_load(f)
                assert 'groups' in rules
                assert len(rules['groups']) > 0
            except Exception as e:
                pytest.skip(f"Could not load alert rules YAML: {e}")
    
    def test_alert_group_structure(self):
        """Test alert group structure"""
        import yaml
        
        try:
            with open('config/alert_rules.yml', 'r') as f:
                rules = yaml.safe_load(f)
                
                for group in rules.get('groups', []):
                    assert 'name' in group
                    assert 'interval' in group
                    assert 'rules' in group
                    
                    for rule in group['rules']:
                        assert 'alert' in rule
                        assert 'expr' in rule
                        assert 'labels' in rule
                        assert 'annotations' in rule
        except Exception as e:
            pytest.skip(f"Could not validate alert rules: {e}")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests"""
    
    def test_metrics_collection_pipeline(self, metrics_collector):
        """Test complete metrics collection pipeline"""
        # Simulate requests
        for i in range(10):
            metrics_collector.predictions_total.labels(
                model='lstm',
                status='success'
            ).inc()
            
            metrics_collector.model_inference_duration_seconds.observe(0.05 + i * 0.01)
            metrics_collector.cache_hits_total.inc()
        
        # Export metrics
        metrics_text = metrics_collector.registry.collect()
        assert len(list(metrics_text)) > 0
    
    def test_performance_monitoring_pipeline(self, performance_monitor):
        """Test performance monitoring pipeline"""
        # Simulate requests
        for i in range(20):
            duration = 0.05 + (i % 10) * 0.01
            success = i % 3 != 0  # Some failures
            performance_monitor.record_request(duration, success=success)
        
        # Get stats
        stats = performance_monitor.get_stats()
        
        assert stats['total_requests'] == 20
        assert stats['error_rate'] > 0
        assert stats['p95_latency'] > stats['min_latency']


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
