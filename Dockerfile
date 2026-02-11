# Multi-stage Dockerfile for Stock Price Prediction API

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-api.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-api.txt


# Stage 2: Runtime
FROM python:3.10-slim

# Set labels
LABEL maintainer="Stock Price Prediction Team"
LABEL description="FastAPI Stock Price Prediction Service"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    APP_HOME="/app"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser && \
    mkdir -p $APP_HOME && \
    chown -R appuser:appuser $APP_HOME

WORKDIR $APP_HOME

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser production/api/ .
COPY --chown=appuser:appuser production/models/ ./models/
COPY --chown=appuser:appuser production/features/ ./features/
COPY --chown=appuser:appuser production/validation/ ./validation/

# Switch to app user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
