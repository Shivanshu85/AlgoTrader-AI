# Stock Price Prediction - Production Grade Platform

A comprehensive, production-ready machine learning platform for stock price prediction using LSTM and attention mechanisms. Built with enterprise-grade architecture, including data pipelines, model training, serving, monitoring, and deployment infrastructure.

## 🎯 Project Goals

Transform a tutorial LSTM model into a production-grade system with:

- ✅ **Robust Data Engineering**: Multi-source data ingestion with quality validation
- ✅ **Production-grade ML**: Reproducible training, evaluation, and serving
- ✅ **Scalable Architecture**: Kubernetes-ready, cloud-agnostic deployment
- ✅ **Comprehensive Monitoring**: Performance, data quality, and business metrics
- ✅ **Complete Testing**: Unit, integration, and performance tests
- ✅ **Enterprise Security**: Authentication, encryption, audit logging
- ✅ **MLOps Best Practices**: Experiment tracking, model registry, versioning

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git
- PostgreSQL 15
- Redis 7

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/example/stock-predictor-prod.git
cd stock-predictor-prod

# Copy environment file
cp .env.example .env

# Set up development environment (installs dependencies)
make setup

# Start Docker services (PostgreSQL, Redis, MLflow, Airflow)
make docker-up

# Verify all services are running
make health-check
```

### 2. Run Training Pipeline

```bash
# Start a training pipeline
make run-training

# Or use Airflow for scheduling
make run-scheduler
make run-webui  # http://localhost:8080
```

### 3. Start Prediction API

```bash
# Development server (with auto-reload)
make run-api

# Server will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 4. Monitor Performance

```bash
# Access MLflow tracking UI
open http://localhost:5000

# Monitor with Prometheus/Grafana
open http://localhost:3000  # Username: admin, Password: from .env
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                         │
│  (Alpha Vantage, IEX Cloud, FMP, Yahoo Finance)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Data Processing Pipeline                       │
│  (Validation, Cleaning, Feature Engineering, Feast)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Model Training & Evaluation                        │
│  (PyTorch LSTM, Attention, Hyperparameter Tuning)             │
├─────────────────────────────────────────────────────────────────┤
│                      MLflow Registry                            │
│  (Experiment Tracking, Model Versioning, A/B Testing)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               Model Serving & Prediction API                    │
│  (FastAPI, Redis Caching, Load Balancing)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         Monitoring, Alerting & Continuous Improvement          │
│  (Prometheus, Grafana, Data Drift Detection)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | Apache Airflow | Workflow scheduling & DAG management |
| **ML Framework** | PyTorch Lightning | Model training & distributed learning |
| **Feature Store** | Feast (upcoming) | Feature management & serving |
| **Model Registry** | MLflow | Experiment tracking & model versioning |
| **API Serving** | FastAPI | REST API for predictions |
| **Caching** | Redis | Feature & prediction caching |
| **Database** | PostgreSQL | Data storage & metadata |
| **Monitoring** | Prometheus + Grafana | Metrics & visualization |
| **Deployment** | Kubernetes | Container orchestration |
| **CI/CD** | GitHub Actions | Automated testing & deployment |

## 📁 Project Structure

```
stock-predictor-prod/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline
├── airflow/
│   ├── dags/                     # Airflow DAGs
│   ├── logs/
│   └── plugins/
├── configs/
│   ├── default.yaml              # Default configuration
│   ├── production.yaml            # Production settings
│   ├── prometheus.yml             # Prometheus config
│   └── grafana/
│       └── provisioning/          # Grafana dashboards
├── data/
│   ├── raw/                       # Original data from APIs
│   ├── processed/                 # Cleaned & validated data
│   └── features/                  # Engineered features
├── docs/
│   ├── architecture/              # System design docs
│   │   └── adr/                  # Architecture Decision Records
│   ├── api/                       # API documentation
│   └── guides/                    # User guides
├── docker/
│   ├── Dockerfile.train           # Training image
│   ├── Dockerfile.serve           # Serving image
│   ├── Dockerfile.dev             # Development image
│   └── Dockerfile.airflow         # Airflow image
├── k8s/
│   ├── deployment.yaml            # K8s deployment
│   ├── service.yaml               # K8s service
│   ├── configmap.yaml             # K8s config
│   └── secrets.yaml               # K8s secrets
├── notebooks/
│   ├── exploration/               # Data exploration
│   └── experiments/               # Model experiments
├── production/
│   ├── data_ingestion/            # Data collection APIs
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── alpha_vantage.py
│   │   ├── iex_cloud.py
│   │   └── validators.py
│   ├── features/                  # Feature engineering
│   │   ├── __init__.py
│   │   ├── technical_indicators.py
│   │   ├── statistical.py
│   │   └── feature_store.py
│   ├── models/                    # Model definitions
│   │   ├── __init__.py
│   │   ├── lstm.py
│   │   ├── attention.py
│   │   └── ensemble.py
│   ├── training/                  # Training pipeline
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── hyperparameter_tuning.py
│   │   └── callbacks.py
│   ├── serving/                   # Prediction serving
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── models_loader.py
│   │   └── cache.py
│   ├── monitoring/                # Monitoring & metrics
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── data_drift.py
│   │   └── alerting.py
│   ├── utils/                     # Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── database.py
│   └── health_check.py            # System health check
├── scripts/
│   ├── init_db.sql                # Database initialization
│   ├── train_model.py             # Training script
│   ├── evaluate_model.py           # Evaluation script
│   └── deploy_model.py            # Deployment script
├── tests/
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── performance/               # Performance tests
├── .env.example                   # Environment template
├── .gitignore
├── .pre-commit-config.yaml        # Pre-commit hooks
├── docker-compose.yml             # Docker services
├── Makefile                       # Development tasks
├── pyproject.toml                 # Project metadata & dependencies
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/example/stock-predictor-prod.git
cd stock-predictor-prod
```

### 2. Set Up Python Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n stock-pred python=3.10
conda activate stock-pred
```

### 3. Install Dependencies

```bash
# Install pyproject dependencies
pip install -e .

# Install dev dependencies for development
pip install -e ".[dev]"

# Run setup (installs pre-commit hooks)
make setup
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys and credentials
nano .env
```

### 5. Initialize Database

```bash
# Start PostgreSQL via Docker
make docker-up

# Wait for PostgreSQL to be ready, then initialize
make db-init
```

## 💻 Usage

### Data Ingestion

```python
from production.data_ingestion import AlphaVantageProvider

provider = AlphaVantageProvider(api_key="your_key")
data = provider.get_daily("AAPL", start_date="2023-01-01")
```

### Feature Engineering

```python
from production.features import TechnicalIndicators
from datetime import datetime, timedelta

indicators = TechnicalIndicators(lookback_period=60)
features = indicators.calculate_all(df)
```

### Model Training

```python
from production.training import Trainer
from production.models import LSTMWithAttention

model = LSTMWithAttention(
    input_size=15,  # Number of features
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

trainer = Trainer(model)
history = trainer.fit(train_loader, val_loader, epochs=50)
```

### Making Predictions

```python
from production.serving import PredictionServer

server = PredictionServer()
prediction = server.predict(
    ticker="AAPL",
    days_ahead=5
)
```

### Via API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "days_ahead": 5,
    "confidence_interval": 0.95
  }'
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run integration tests
make test-integration

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/unit/test_feature_engineering.py -v
```

## 🔧 Development

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type checking
make type-check

# Security scanning
make security
```

### Start Development Containers

```bash
# Start all services
make docker-up

# View logs
make docker-logs

# Access services
# PostgreSQL: localhost:5432
# Redis: localhost:6379
# MLflow: http://localhost:5000
# Airflow: http://localhost:8080
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
# Adminer: http://localhost:8081
```

### Running Jupyter Notebook

```bash
make notebook

# Open http://localhost:8888
```

## 🚀 Deployment

### Docker Build

```bash
# Build training image
docker build -f docker/Dockerfile.train -t stock-predictor:latest .

# Build API image
docker build -f docker/Dockerfile.serve -t stock-predictor-api:latest .
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl logs -f <pod-name>
```

### Environment Variables

```bash
# Production deployment requires these environment variables
export POSTGRES_HOST=prod-db.example.com
export POSTGRES_PASSWORD=<secure-password>
export REDIS_HOST=prod-redis.example.com
export MLFLOW_TRACKING_URI=https://mlflow.example.com
export API_KEY_SECRET=<secure-api-key>
```

## 📊 Monitoring

### MLflow Experiment Tracking

```bash
make run-tracker  # http://localhost:5000
```

### Metrics & Visualization

```bash
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### Application Logs

```bash
# View logs
docker-compose logs -f api

# Or in development
tail -f logs/app.log
```

## 📚 Documentation

Build and view complete documentation:

```bash
make docs
make docs-serve
```

Navigate to `http://localhost:8000` in your browser.

See the [docs/](docs/) folder for:
- Architecture decisions ([docs/architecture/adr/](docs/architecture/adr/))
- System design ([docs/architecture/system_design.md](docs/architecture/system_design.md))
- API documentation ([docs/api/](docs/api/))
- User guides ([docs/guides/](docs/guides/))

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Resources

- [PyTorch Documentation](https://pytorch.org/docs)
- [MLflow Documentation](https://mlflow.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Airflow Documentation](https://airflow.apache.org/docs)
- [Prometheus Documentation](https://prometheus.io/docs)

## 📧 Support

For issues, questions, or suggestions:
- Open an [Issue](https://github.com/example/stock-predictor-prod/issues)
- Start a [Discussion](https://github.com/example/stock-predictor-prod/discussions)
- Contact: team@example.com

---

**Built with ❤️ by the ML Engineering Team**
