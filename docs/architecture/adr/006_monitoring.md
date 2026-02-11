# ADR 006: Monitoring & Observability - Prometheus + Grafana vs Datadog vs New Relic

**Status:** ACCEPTED  
**Date:** February 2026  
**ADR Number:** 006  

---

## Decision

**SELECTED: Prometheus + Grafana (Open-source stack)**

### Pros
- ✅ Zero licensing cost
- ✅ Industry standard for Kubernetes
- ✅ Time-series database optimized for metrics
- ✅ Flexible querying language (PromQL)
- ✅ Grafana for visualization (gorgeous dashboards)
- ✅ AlertManager for alerting
- ✅ Large community

### Implementation

```yaml
# configs/prometheus.yml

global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['localhost:8000']
  
  - job_name: 'kubernetes'
    kubernetes_sd_configs:
      - role: pod
```

```python
# production/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Metrics
prediction_counter = Counter(
    'predictions_total',
    'Total predictions',
    ['ticker']
)

latency_histogram = Histogram(
    'prediction_latency_seconds',
    'Prediction latency',
    buckets=(0.01, 0.05, 0.1, 0.5)
)

cache_hit_gauge = Gauge(
    'cache_hit_rate',
    'Cache hit rate (0-1)'
)

@app.post("/predict")
async def predict(request):
    with latency_histogram.time():
        result = model.predict(request.ticker)
    prediction_counter.labels(ticker=request.ticker).inc()
    return result
```

---

**Status:** ✅ ACCEPTED
