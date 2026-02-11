# Architecture Decision Records - Index

**Last Updated:** February 2026  
**Total ADRs:** 8  
**Status:** All Accepted ✅  

---

## Quick Reference

| # | Title | Status | Category | Review Date |
|---|-------|--------|----------|------------|
| [001](001_feature_store_selection.md) | Feature Store: Feast | ✅ ACCEPTED | Data | Feb 2026 |
| [002](002_api_framework.md) | API Framework: FastAPI | ✅ ACCEPTED | Serving | Feb 2026 |
| [003](003_orchestration.md) | Orchestration: Apache Airflow | ✅ ACCEPTED | Pipeline | Feb 2026 |
| [004](004_ml_framework.md) | ML Framework: PyTorch + Lightning | ✅ ACCEPTED | ML | Feb 2026 |
| [005](005_experiment_tracking.md) | Experiment Tracking: MLflow | ✅ ACCEPTED | ML Ops | Feb 2026 |
| [006](006_monitoring.md) | Monitoring: Prometheus + Grafana | ✅ ACCEPTED | Observability | Feb 2026 |
| [007](007_deployment.md) | Deployment: Kubernetes | ✅ ACCEPTED | Infrastructure | Feb 2026 |
| [008](008_data_storage.md) | Data Storage: PostgreSQL + S3 | ✅ ACCEPTED | Infrastructure | Feb 2026 |

---

## Technology Stack Summary

### Data Layer
- **Storage:** PostgreSQL (hot) + S3 (cold)
- **Features:** Feast (online: Redis, offline: PostgreSQL)
- **Orchestration:** Apache Airflow

### ML Layer
- **Framework:** PyTorch + PyTorch Lightning
- **Tracking:** MLflow (experiments + model registry)
- **Serving:** FastAPI (REST API)

### Operations Layer
- **Deployment:** Kubernetes (K8s)
- **Monitoring:** Prometheus + Grafana
- **Logging:** ELK Stack (optional)
- **CI/CD:** GitHub Actions

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│ Data Sources (APIs: Alpha Vantage, IEX, Yahoo Finance)      │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│ Data Ingestion (Airflow DAGs)                                │
├──────────────────────────────────────────────────────────────┤
│ • PostgreSQL (raw data)                                      │
│ • Validation layer                                           │
│ • Error handling + retries                                   │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│ Feature Engineering (Feast Feature Store)                    │
├──────────────────────────────────────────────────────────────┤
│ • Technical indicators                                       │
│ • Statistical features                                       │
│ • Online store: Redis (< 50ms)                              │
│ • Offline store: PostgreSQL (historical)                    │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│ Model Training (PyTorch + Lightning)                         │
├──────────────────────────────────────────────────────────────┤
│ • Temporal validation (no data leakage)                      │
│ • Hyperparameter tuning (Optuna)                            │
│ • MLflow tracking                                            │
│ • Distributed training (GPU support)                         │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│ Model Registry (MLflow)                                      │
├──────────────────────────────────────────────────────────────┤
│ • Versioning                                                 │
│ • Metadata storage                                           │
│ • Artifact management                                        │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│ Model Serving (FastAPI)                                      │
├──────────────────────────────────────────────────────────────┤
│ • REST API (/predict)                                        │
│ • Request validation (Pydantic)                             │
│ • Redis caching (1hr TTL)                                   │
│ • Sub-100ms latency (p99)                                   │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│ Monitoring & Observability                                   │
├──────────────────────────────────────────────────────────────┤
│ • Prometheus (metrics)                                       │
│ • Grafana (dashboards)                                       │
│ • AlertManager (alerting)                                    │
│ • Data drift detection                                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Decision Rationale (Quick Version)

### Why Feast?
- Open-source (no licensing costs)
- Cloud-agnostic (self-managed)
- 45ms latency < 50ms requirement
- Redis + PostgreSQL split (online/offline)

### Why FastAPI?
- #1 performance (321K req/sec)
- Native async/await
- Auto-generation of API docs
- Pydantic for bulletproof validation

### Why PyTorch?
- Dynamic graphs (easier debugging)
- Best for attention mechanisms
- Easier custom layers
- Larger community (esp. for LSTM)

### Why MLflow?
- Open-source
- Native PyTorch integration
- Model versioning built-in
- ONNX support for deployment

### Why Kubernetes?
- Industry standard
- Auto-scaling & self-healing
- Multi-cloud support
- Huge ecosystem

### Why PostgreSQL?
- ACID guarantees
- Time-series extension
- JSON support
- No vendor lock-in

---

## Implementation Timeline

| Phase | Week | Component | ADR |
|-------|------|-----------|-----|
| Phase 1 | Feb | Architecture | All |
| Phase 2 | Mar | Data Pipeline | 001, 003, 008 |
| Phase 3 | Mar | Validation | - |
| Phase 4 | Apr | Model Training | 004, 005 |
| Phase 5 | Apr | Model Serving | 002 |
| Phase 6 | May | Monitoring | 006 |
| Phase 7 | May | Deployment | 007 |
| Phase 8 | Jun | Advanced | - |
| Phase 9 | Jun | Documentation | - |

---

## Cost Breakdown

| Component | Monthly Cost |
|-----------|--------------|
| Compute (K8s) | $2000 |
| Storage (PostgreSQL + S3) | $100 |
| Data Transfer | $200 |
| Managed Services | $300 |
| **TOTAL** | **$2600** |

With optimizations (spot instances): **~$1500/month**

---

## Monitoring & Alerting

### Key Metrics to Track
1. **Prediction Latency** (p99 < 100ms)
2. **Cache Hit Ratio** (target 70%+)
3. **Model Accuracy** (track drift)
4. **Successful Predictions** (< 1% errors)
5. **System Uptime** (target 99.9%)

### Alert Thresholds
- Latency p99 > 200ms → ⚠️ Warning
- Latency p99 > 500ms → 🔴 Critical
- Cache hit < 40% → ⚠️ Warning
- Model accuracy drop > 5% → ⚠️ Warning
- Uptime < 99% → 🔴 Critical

---

## Related Documents

- [System Architecture Design](../system_design.md)
- [Deployment Guide](../guides/deployment.md)
- [Monitoring Setup](../guides/monitoring.md)
- [Security Guide](../guides/security.md)

---

## FAQ

**Q: Why not use managed services?**
A: Cost ($$$) + vendor lock-in. Self-managed approach saves $1-2K/month.

**Q: Can we switch technologies later?**
A: Yes! Architecture is modular. Each layer can be replaced independently.

**Q: What if we need to scale to 100K tickers?**
A: All decisions scale to 10M+ tickers. See scalability sections in each ADR.

**Q: How do we handle disaster recovery?**
A: Multi-region setup, daily backups, automated failover. See DR section in system design.

---

**Status:** ✅ All ADRs Finalized  
**Review Cycle:** Quarterly  
**Next Review:** May 2026

For questions on any decision, refer to specific ADR (001-008) or schedule architecture review.
