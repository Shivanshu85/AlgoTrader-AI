# ADR 005: Model Registry & Experiment Tracking - MLflow vs Weights&Biases vs Neptune

**Status:** ACCEPTED  
**Date:** February 2026  
**ADR Number:** 005  

---

## Decision

**SELECTED: MLflow**

### Pros
- ✅ Open-source, completely free
- ✅ Language-agnostic (Python, R, Java)
- ✅ Seamless PyTorch integration
- ✅ Model registry with versioning
- ✅ Easy deployment
- ✅ Built-in model serving
- ✅ ONNX support

### Implementation

```python
# production/tracking/mlflow_utils.py

import mlflow
from mlflow.pytorch import log_model

def log_experiment(config, model, metrics):
    mlflow.set_experiment("stock_prediction")
    
    with mlflow.start_run():
        # Log params
        mlflow.log_params(config)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        log_model(model, "model")
        
        # Log artifacts
        mlflow.log_artifact("plots/training_history.png")
        
        run_id = mlflow.active_run().info.run_id
        return run_id
```

---

**Status:** ✅ ACCEPTED
