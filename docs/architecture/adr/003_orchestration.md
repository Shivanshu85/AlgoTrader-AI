# ADR 003: Orchestration Framework - Apache Airflow vs Prefect vs Dagster

**Status:** ACCEPTED  
**Date:** February 2026  
**ADR Number:** 003  

---

## Context

We need a workflow orchestration platform to:
- Schedule daily model training
- Monitor data ingestion pipelines
- Handle dependencies and retries
- Provide visibility into job status
- Scale to 100+ DAGs
- Support dynamic pipeline generation

---

## Options Evaluated

### Option 1: Apache Airflow (SELECTED)

**Pros:**
- ✅ Battle-tested (7+ years production)
- ✅ Largest community (GCP, Airbnb, Uber use it)
- ✅ Rich UI for monitoring
- ✅ Extensive integrations
- ✅ Stable API
- ✅ Scalable (we run 100+ DAGs)

**Cons:**
- ⚠️ Steep learning curve initially
- ⚠️ Boilerplate-heavy
- ⚠️ Need strong DevOps skills

### Option 2: Prefect

**Cons:**
- ❌ Newer, community support smaller
- ❌ More opinionated design
- ❌ Vendor pricing becoming relevant ($$$)

### Option 3: Dagster

**Cons:**
- ❌ Newer, ecosystem less mature
- ❌ Heavier resource requirements
- ❌ Steeper learning curve

---

## Decision

**SELECTED: Apache Airflow**

### Rationale

1. **Maturity:** 7+ years in production, stable
2. **Community:** Largest ecosystem, most tutorials
3. **Scalability:** Proven to handle 1000+ DAGs
4. **Cost:** Open-source, no licensing
5. **Learning:** Rich documentation and examples

---

## Implementation

```python
# airflow/dags/daily_training.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml_team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['ml-alerts@example.com'],
}

dag = DAG(
    'daily_model_training',
    default_args=default_args,
    description='Daily LSTM model training',
    schedule_interval='0 2 * * *',  # 2 AM UTC
    start_date=datetime(2026, 2, 1),
    catchup=False,
)

def fetch_data_task(**context):
    from production.data_ingestion import DataIngestionPipeline
    pipeline = DataIngestionPipeline()
    pipeline.run()

def prepare_features_task(**context):
    from production.features import FeatureEngineering
    fe = FeatureEngineering()
    fe.run()

def train_model_task(**context):
    from production.training import train
    train()

def evaluate_model_task(**context):
    from production.training import evaluate
    evaluate()

# Define tasks
fetch_data = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_task,
    dag=dag,
)

prepare_features = PythonOperator(
    task_id='prepare_features',
    python_callable=prepare_features_task,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag,
)

# Set dependencies
fetch_data >> prepare_features >> train_model >> evaluate_model
```

---

## Consequences

✅ Proven reliability  
✅ Rich ecosystem  
✅ Strong community support  
✅ Excellent monitoring UI  

---

**Status:** ✅ ACCEPTED  
**Implementation:** Phase 2 (Data Engineering)
