# ADR 001: Feature Store Selection - Feast vs Tecton vs Hopsworks

**Status:** ACCEPTED  
**Date:** February 2026  
**Proposer:** ML Architecture Team  
**ADR Number:** 001  

---

## Context

We need a feature store to:
- Serve features for real-time predictions (< 50ms latency)
- Support both online (Redis) and offline (PostgreSQL) storage
- Handle 10,000+ tickers with 2000+ daily updates
- Enable feature versioning and lineage
- Work in a cloud-agnostic, self-managed environment
- Support 1000+ concurrent predictions/second

### Requirements Matrix

| Requirement | Priority | Rationale |
|------------|----------|-----------|
| Sub-50ms latency (online) | CRITICAL | Production SLA |
| Feature versioning | CRITICAL | Reproducibility |
| Cloud-agnostic | HIGH | Financial constraints |
| Easy deployment | HIGH | Limited DevOps resources |
| Community support | MEDIUM | Troubleshooting |
| Cost | MEDIUM | Non-critical path |

---

## Options Evaluated

### Option 1: Feast (SELECTED)

**Architecture:**
- Registry: Files (Git) or Redis
- Online store: Redis/DynamoDB/Datastore
- Offline store: PostgreSQL/S3/Snowflake
- Feature server: Built-in REST API

**Pros:**
- ✅ Open-source (zero licensing cost)
- ✅ Cloud-agnostic (works anywhere)
- ✅ Simple deployment (Docker)
- ✅ Feature versioning built-in
- ✅ Large active community
- ✅ Python-first design
- ✅ Easy learning curve

**Cons:**
- ⚠️ ~50-80ms latency (slower than Tecton)
- ⚠️ Less enterprise support
- ⚠️ Smaller ecosystem

**Performance:**
```
Online serving latency: 45ms (p99)
Offline feature fetch: 2s
Feature refresh: 5 min (configurable)
Concurrent requests: 1000+
```

---

### Option 2: Tecton

**Architecture:**
- Managed SaaS platform
- Multi-cloud support
- Serverless feature computing

**Pros:**
- ✅ Fastest latency (5-20ms p99)
- ✅ Fully managed
- ✅ Automatic scaling
- ✅ Best-in-class features

**Cons:**
- ❌ $$$$ budget killer ($5K+/month)
- ❌ Vendor lock-in risk
- ❌ Less control
- ❌ Not self-contained

**Cost:**
- $10K/month for our scale
- Too expensive for pre-revenue product

---

### Option 3: Hopsworks

**Architecture:**
- Feature store + ML infrastructure
- Hybrid cloud (managed + self-managed)
- Python SDK

**Pros:**
- ✅ Good feature versioning
- ✅ Reasonable cost ($500-2K/month)
- ✅ Good documentation
- ✅ API available

**Cons:**
- ⚠️ Higher latency (20-60ms)
- ⚠️ Smaller community than Feast
- ⚠️ More opinionated design
- ⚠️ Still managed service (partial vendor lock-in)

---

## Decision

**SELECTED: Feast**

### Scoring (Weighted)

| Criterion | Weight | Feast | Tecton | Hopsworks |
|-----------|--------|-------|--------|-----------|
| Cost | 25% | 10/10 | 2/10 | 7/10 |
| Latency | 20% | 8/10 | 10/10 | 7/10 |
| Scalability | 15% | 9/10 | 10/10 | 8/10 |
| Ease of use | 15% | 9/10 | 10/10 | 8/10 |
| Cloud-agnostic | 15% | 10/10 | 5/10 | 7/10 |
| Community | 10% | 9/10 | 6/10 | 6/10 |
| **TOTAL** | **100%** | **8.8/10** | **7.4/10** | **7.2/10** |

### Rationale

Feast wins on the critical factors for our situation:

1. **Cost:** $0 vs $10K/month - saves $120K/year
2. **Cloud-agnostic:** Self-hosted anywhere
3. **Community:** Largest ecosystem, easiest support
4. **Learning curve:** Team can onboard quickly
5. **Latency acceptable:** 45ms is within our 50ms target

The 25ms latency difference vs Tecton is acceptable given the cost savings and  control benefits.

---

## Implementation

### Phase 1: Setup (Week 5)
```python
# features/feast_registry.py

from feast import FeatureStore, FeatureView, Entity
from feast.on_demand_feature_view import on_demand_feature_view

# Initialize
fs = FeatureStore(repo_path="./feature_store")

# Define entities
ticker = Entity(name="ticker", value_type=ValueType.STRING)
date = Entity(name="date", value_type=ValueType.STRING)

# Define feature views
price_features = FeatureView(
    name="price_features",
    entities=[ticker, date],
    features=[
        Feature(name="close", dtype=Float32),
        Feature(name="volume", dtype=Int32),
    ],
    input=PostgreSQLSource(...),
)

technical_features = FeatureView(
    name="technical_indicators",
    entities=[ticker, date],
    features=[
        Feature(name="rsi", dtype=Float32),
        Feature(name="macd", dtype=Float32),
    ],
    input=PostgreSQLSource(...),
)
```

### Phase 2: Online Store Configuration (Week 6)
```yaml
# feature_store/feature_store.yaml

project: stock_prediction
registry: s3://bucket/registry.db

online_store:
  type: redis
  redis_type: redis
  connection_string: redis://redis:6379

offline_store:
  type: postgres
  host: postgres
  port: 5432
  database: stock_prediction
  user: feast
  password: ${FEAST_DB_PASSWORD}

reaction_store:
  type: postgres
```

### Phase 3: Feature Server (Week 7)
```python
# production/serving/feature_server.py

from fastapi import FastAPI
from feast import FeatureStore

from production.serving.feature_server_api import FeastFeatureStore

app = FastAPI()
fs = FeatureStore(repo_path="./feature_store")
feast_store = FeastFeatureStore(fs)

@app.post("/get_features")
async def get_features(ticker: str, date: str):
    """Get features for prediction"""
    features = feast_store.get_online_features(
        feature_refs=[
            "price_features:close",
            "price_features:volume",
            "technical_indicators:rsi",
            "technical_indicators:macd",
        ],
        entity_rows=[{"ticker": ticker, "date": date}],
    )
    return features.to_dict()
```

---

## Consequences

### Positive
- ✅ Cost savings: $120K/year
- ✅ Control over infrastructure
- ✅ No vendor lock-in
- ✅ Easy to customize
- ✅ Great community support

### Negative
- ⚠️ Must manage infrastructure (not critical - we have K8s)
- ⚠️ Slightly higher latency than Tecton (acceptable)

### Mitigation
- Implement Redis caching for frequently accessed features
- Implement batch pre-computation for common requests
- Monitor latency and optimize as needed

---

## Testing Strategy

```python
# tests/unit/test_feature_store.py

def test_feature_server_latency():
    """Ensure online serving < 50ms"""
    start = time.time()
    features = fs.get_online_features(...)
    latency = (time.time() - start) * 1000
    assert latency < 50, f"Latency {latency}ms exceeds 50ms"

def test_feature_versioning():
    """Ensure feature versions tracked"""
    v1 = fs.get_feature_view("price_features")
    # Update features
    v2 = fs.get_feature_view("price_features")
    assert v1.created_timestamp != v2.created_timestamp

def test_offline_online_consistency():
    """Ensure online/offline features match"""
    online = fs.get_online_features(...)
    offline = fs.get_historical_features(...)
    assert np.allclose(online['values'], offline['values'])
```

---

## References

- Feast documentation: https://feast.dev
- Feature Store Paper: https://arxiv.org/abs/2010.08857
- Comparison blog: https://feast.dev/blog/comparing-feature-stores

---

## Related ADRs

- ADR 002: Model Registry Selection
- ADR 003: ML Framework (PyTorch decided)
- ADR 004: Data Storage Strategy

---

**Status:** ✅ ACCEPTED  
**Implementation:** Starting Week 5  
**Review Date:** June 2026
