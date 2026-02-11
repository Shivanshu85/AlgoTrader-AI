"""
Feast Feature Store Configuration and Feature Definitions

Defines all features for the stock price prediction model:
- Technical indicators (RSI, MACD, SMA, etc.)
- Market context features
- Target variables
"""

from datetime import timedelta
from feast import (
    FeatureStore,
    FeatureView,
    Entity,
    Field,
    FileSource,
    PushSource,
    SQLSource,
    FeatureService,
    OnDemandFeatureView,
    RequestSource
)
from feast.infra.offline_store.postgres import PostgreSQLOfflineStoreConfig
from feast.infra.online_store.redis import RedisOnlineStoreConfig
from feast.types import Float32, Int64, String
from datetime import datetime
import pandas as pd


# ============================================================================
# FEATURE STORE CONFIGURATION
# ============================================================================

# Define the offline (batch) store - PostgreSQL
offline_store = PostgreSQLOfflineStoreConfig(
    host="localhost",
    port=5432,
    database="stock_prediction",
    db_schema="public",
    user="app_user",
    password="secure_password",
)

# Define the online (real-time) store - Redis
online_store = RedisOnlineStoreConfig(
    connection_string="redis://localhost:6379",
)

# Initialize feature store
fs = FeatureStore(
    repo_path=".",
    project="stock_prediction",
    registry="s3://stock-prediction-bucket/feast-registry.db",
    offline_store=offline_store,
    online_store=online_store,
)


# ============================================================================
# ENTITIES
# ============================================================================

# Ticker entity (stock symbol)
ticker = Entity(
    name="ticker",
    description="Stock ticker symbol",
    value_type="STRING",
    tags={"team": "data-engineering"},
)


# ============================================================================
# DATA SOURCES
# ============================================================================

# Raw market data from PostgreSQL
raw_market_source = SQLSource(
    table="raw_market_data",
    timestamp_field="ingested_at",
    created_timestamp_column="ingested_at",
    path="postgresql://localhost/stock_prediction",
)

# Technical indicators features
technical_indicators_source = SQLSource(
    table="feature_technical_indicators",
    timestamp_field="computed_at",
    created_timestamp_column="computed_at",
    path="postgresql://localhost/stock_prediction",
)

# Market context features
market_context_source = SQLSource(
    table="feature_market_context",
    timestamp_field="computed_at",
    created_timestamp_column="computed_at",
    path="postgresql://localhost/stock_prediction",
)

# Target variables
target_source = SQLSource(
    table="target_daily",
    timestamp_field="computed_at",
    created_timestamp_column="computed_at",
    path="postgresql://localhost/stock_prediction",
)


# ============================================================================
# FEATURE VIEWS - TECHNICAL INDICATORS
# ============================================================================

technical_indicators_fv = FeatureView(
    name="technical_indicators",
    entities=[ticker],
    ttl=timedelta(days=180),  # Keep 6 months of history
    schema=[
        Field(name="rsi_14", dtype=Float32, description="RSI 14-day"),
        Field(name="macd_line", dtype=Float32, description="MACD line"),
        Field(name="macd_signal", dtype=Float32, description="MACD signal"),
        Field(name="macd_histogram", dtype=Float32, description="MACD histogram"),
        Field(name="sma_20", dtype=Float32, description="SMA 20-day"),
        Field(name="sma_50", dtype=Float32, description="SMA 50-day"),
        Field(name="sma_200", dtype=Float32, description="SMA 200-day"),
        Field(name="ema_12", dtype=Float32, description="EMA 12-day"),
        Field(name="ema_26", dtype=Float32, description="EMA 26-day"),
        Field(name="atr_14", dtype=Float32, description="ATR 14-day"),
        Field(name="bollinger_upper", dtype=Float32, description="Bollinger upper band"),
        Field(name="bollinger_middle", dtype=Float32, description="Bollinger middle band"),
        Field(name="bollinger_lower", dtype=Float32, description="Bollinger lower band"),
        Field(name="obv", dtype=Int64, description="On-Balance Volume"),
        Field(name="volume_sma", dtype=Int64, description="Volume SMA"),
        Field(name="price_range", dtype=Float32, description="High - Low"),
        Field(name="price_range_pct", dtype=Float32, description="Price range %"),
        Field(name="upper_shadow", dtype=Float32, description="Upper shadow %"),
        Field(name="lower_shadow", dtype=Float32, description="Lower shadow %"),
    ],
    source=technical_indicators_source,
    tags={
        "team": "data-science",
        "feature_type": "technical_indicators",
    },
)

# ============================================================================
# FEATURE VIEWS - MARKET CONTEXT
# ============================================================================

market_context_fv = FeatureView(
    name="market_context",
    entities=[ticker],
    ttl=timedelta(days=365),
    schema=[
        Field(name="market_sentiment", dtype=String, description="Market sentiment"),
        Field(name="sector_momentum", dtype=Float32, description="Sector momentum"),
        Field(name="sector_volatility", dtype=Float32, description="Sector volatility"),
        Field(name="correlation_sp500", dtype=Float32, description="Correlation with S&P 500"),
        Field(name="correlation_sector", dtype=Float32, description="Correlation with sector"),
        Field(name="correlation_tech", dtype=Float32, description="Correlation with tech index"),
        Field(name="market_volatility", dtype=Float32, description="Market volatility (VIX-like)"),
        Field(name="market_trend", dtype=String, description="Market trend direction"),
        Field(name="market_trend_strength", dtype=Float32, description="Market trend strength"),
    ],
    source=market_context_source,
    tags={
        "team": "data-science",
        "feature_type": "market_context",
    },
)

# ============================================================================
# ON-DEMAND FEATURE VIEWS
# ============================================================================

# Compute features on-the-fly during prediction
request_source = RequestSource(
    name="prediction_request",
    schema=[
        Field(name="prediction_date", dtype=String),
        Field(name="current_price", dtype=Float32),
    ],
)

# Example: compute price momentum on-the-fly
price_momentum_fv = OnDemandFeatureView(
    name="price_momentum",
    sources=[technical_indicators_fv, request_source],
    schema=[
        Field(name="momentum_5d", dtype=Float32, description="5-day momentum"),
        Field(name="momentum_20d", dtype=Float32, description="20-day momentum"),
    ],
    udf=lambda df: pd.DataFrame({
        "momentum_5d": df["sma_20"] / df["current_price"] - 1,
        "momentum_20d": df["sma_50"] / df["current_price"] - 1,
    }),
)

# ============================================================================
# TARGET FEATURES
# ============================================================================

target_fv = FeatureView(
    name="target_next_day",
    entities=[ticker],
    ttl=timedelta(days=365),
    schema=[
        Field(name="target_return", dtype=Float32, description="Next day return"),
        Field(name="target_binary", dtype=Int64, description="1 if up, 0 if down"),
        Field(name="target_direction", dtype=String, description="Direction: up/down/flat"),
        Field(name="confidence", dtype=Float32, description="Target confidence"),
    ],
    source=target_source,
    tags={
        "team": "data-science",
        "feature_type": "target",
    },
)


# ============================================================================
# FEATURE SERVICES
# ============================================================================

# Feature service for training - includes all features needed for model training
training_feature_service = FeatureService(
    name="training_features",
    features=[
        technical_indicators_fv,
        market_context_fv,
        target_fv,
    ],
    tags={
        "stage": "training",
        "description": "Features for model training",
    },
)

# Feature service for inference - excludes target variable
inference_feature_service = FeatureService(
    name="inference_features",
    features=[
        technical_indicators_fv,
        market_context_fv,
        price_momentum_fv,
    ],
    tags={
        "stage": "inference",
        "description": "Features for real-time inference",
    },
)

# Feature service for backtesting
backtesting_feature_service = FeatureService(
    name="backtesting_features",
    features=[
        technical_indicators_fv,
        market_context_fv,
        target_fv,
    ],
    tags={
        "stage": "backtesting",
        "description": "Features for model backtesting",
    },
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_feature_store():
    """Get initialized feature store instance"""
    return fs


def get_historical_features(
    tickers: list,
    start_date: datetime,
    end_date: datetime,
    feature_service_name: str = "training_features"
):
    """
    Fetch historical features for specified tickers and date range
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        feature_service_name: Name of the feature service to use
        
    Returns:
        Pandas DataFrame with features
    """
    # Create entity dataframe
    entity_df = pd.DataFrame({
        "ticker": tickers,
        "event_timestamp": pd.date_range(start=start_date, end=end_date, freq="D"),
    })
    
    # Get feature service
    feature_service = fs.get_feature_service(feature_service_name)
    
    # Fetch features
    features = fs.get_historical_features(
        entity_df=entity_df,
        features=feature_service.features,
    ).to_df()
    
    return features


def get_online_features(
    tickers: list,
    feature_service_name: str = "inference_features"
):
    """
    Fetch online (real-time) features for inference
    
    Args:
        tickers: List of ticker symbols
        feature_service_name: Name of the feature service to use
        
    Returns:
        Dictionary of ticker -> features mapping
    """
    # Get feature service
    feature_service = fs.get_feature_service(feature_service_name)
    
    # Fetch online features
    features_dict = fs.get_online_features(
        features=feature_service.features,
        entity_rows=[{"ticker": ticker} for ticker in tickers],
    ).to_dict()
    
    return features_dict


def push_features_to_online_store(
    ticker: str,
    features: dict,
    event_timestamp: datetime = None
):
    """
    Push features to online store for real-time inference
    
    Args:
        ticker: Stock ticker symbol
        features: Dictionary of feature_name -> value
        event_timestamp: Timestamp for the features (defaults to now)
    """
    if event_timestamp is None:
        event_timestamp = datetime.now()
    
    # Create feature rows
    features_data = {
        "ticker": [ticker],
        "event_timestamp": [event_timestamp],
        **{k: [v] for k, v in features.items()}
    }
    
    features_df = pd.DataFrame(features_data)
    
    # Push to online store
    fs.push("technical_indicators", features_df)


def list_features():
    """List all available features in the store"""
    features = []
    
    for fv in [technical_indicators_fv, market_context_fv, target_fv]:
        for field in fv.schema:
            features.append({
                "feature_view": fv.name,
                "feature": field.name,
                "type": str(field.dtype),
                "description": field.description,
            })
    
    return pd.DataFrame(features)


# ============================================================================
# FEATURE REGISTRY
# ============================================================================

# Register all feature views with the store
def register_features():
    """Register all features with the feature store"""
    fs.apply([ticker, technical_indicators_fv, market_context_fv, target_fv])
    
    # Create feature services
    fs.apply([
        training_feature_service,
        inference_feature_service,
        backtesting_feature_service,
    ])
    
    print("All features registered successfully!")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Register features
    register_features()
    
    # List available features
    print("Available features:")
    print(list_features())
    
    # Example: Get historical features for training
    # from datetime import datetime, timedelta
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=365)
    # 
    # training_features = get_historical_features(
    #     tickers=["AAPL", "GOOGL", "MSFT"],
    #     start_date=start_date,
    #     end_date=end_date,
    #     feature_service_name="training_features"
    # )
    # print(training_features.head())
