-- Production Stock Price Prediction Database Schema
-- PostgreSQL 13+
-- Time-series optimized for stock market data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- 1. RAW DATA TABLES (Source of Truth)
-- ============================================================================

-- Raw OHLCV data from market APIs
CREATE TABLE IF NOT EXISTS raw_market_data (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 4) NOT NULL,
    high DECIMAL(10, 4) NOT NULL,
    low DECIMAL(10, 4) NOT NULL,
    close DECIMAL(10, 4) NOT NULL,
    volume BIGINT NOT NULL,
    adj_close DECIMAL(10, 4),
    source VARCHAR(50) NOT NULL,  -- 'yfinance', 'polygon', 'alpha_vantage'
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date, source)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('raw_market_data', 'date', if_not_exists => TRUE);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_raw_market_data_ticker_date 
ON raw_market_data(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_raw_market_data_ingested_at 
ON raw_market_data(ingested_at DESC);

-- ============================================================================
-- 2. VALIDATION & QUALITY TABLES
-- ============================================================================

-- Data quality metrics
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    missing_ohlcv BOOLEAN DEFAULT FALSE,
    price_outlier BOOLEAN DEFAULT FALSE,
    volume_outlier BOOLEAN DEFAULT FALSE,
    data_gap BOOLEAN DEFAULT FALSE,
    quality_score DECIMAL(3, 2),
    validation_errors JSONB,
    validated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker, date) REFERENCES raw_market_data(ticker, date) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_data_quality_ticker_date 
ON data_quality_metrics(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_score 
ON data_quality_metrics(quality_score);

-- Pipeline execution logs
CREATE TABLE IF NOT EXISTS pipeline_execution_log (
    id BIGSERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100) NOT NULL,
    execution_start TIMESTAMP NOT NULL,
    execution_end TIMESTAMP,
    status VARCHAR(20) NOT NULL,  -- 'running', 'success', 'failed'
    records_processed INT,
    records_failed INT,
    error_message TEXT,
    dag_run_id VARCHAR(100),
    task_instance_key_str VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pipeline_exec_name_status 
ON pipeline_execution_log(pipeline_name, status);

-- ============================================================================
-- 3. PROCESSED/CLEANED DATA TABLES
-- ============================================================================

-- Cleaned and validated OHLCV data
CREATE TABLE IF NOT EXISTS processed_market_data (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 4) NOT NULL,
    high DECIMAL(10, 4) NOT NULL,
    low DECIMAL(10, 4) NOT NULL,
    close DECIMAL(10, 4) NOT NULL,
    volume BIGINT NOT NULL,
    adj_close DECIMAL(10, 4),
    returns DECIMAL(8, 6),  -- (close - prev_close) / prev_close
    log_returns DECIMAL(8, 6),  -- log(close / prev_close)
    cleaning_method VARCHAR(50),  -- 'raw', 'interpolation', 'forward_fill'
    processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_version INT NOT NULL DEFAULT 1,
    UNIQUE(ticker, date)
);

SELECT create_hypertable('processed_market_data', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_processed_market_ticker_date 
ON processed_market_data(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_processed_market_processed_at 
ON processed_market_data(processed_at DESC);

-- ============================================================================
-- 4. FEATURE STORE TABLES (For Feast integration)
-- ============================================================================

-- Technical indicators (computed features)
CREATE TABLE IF NOT EXISTS feature_technical_indicators (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    -- Momentum indicators
    rsi_14 DECIMAL(5, 2),  -- Relative Strength Index (14-day)
    macd_line DECIMAL(8, 6),  -- MACD line
    macd_signal DECIMAL(8, 6),  -- MACD signal line
    macd_histogram DECIMAL(8, 6),  -- MACD histogram
    -- Trend indicators
    sma_20 DECIMAL(10, 4),  -- Simple Moving Average (20-day)
    sma_50 DECIMAL(10, 4),  -- Simple Moving Average (50-day)
    sma_200 DECIMAL(10, 4),  -- Simple Moving Average (200-day)
    ema_12 DECIMAL(10, 4),  -- Exponential Moving Average (12-day)
    ema_26 DECIMAL(10, 4),  -- Exponential Moving Average (26-day)
    -- Volatility indicators
    atr_14 DECIMAL(8, 4),  -- Average True Range (14-day)
    bollinger_upper DECIMAL(10, 4),  -- Bollinger Band upper
    bollinger_middle DECIMAL(10, 4),  -- Bollinger Band middle
    bollinger_lower DECIMAL(10, 4),  -- Bollinger Band lower
    -- Volume indicators
    obv BIGINT,  -- On-Balance Volume
    volume_sma BIGINT,  -- Volume Simple Moving Average
    -- Other features
    price_range DECIMAL(8, 4),  -- (high - low)
    price_range_pct DECIMAL(5, 2),  -- (high - low) / close * 100
    upper_shadow DECIMAL(5, 2),  -- (high - close) / close * 100
    lower_shadow DECIMAL(5, 2),  -- (close - low) / close * 100
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    feature_version INT NOT NULL DEFAULT 1,
    UNIQUE(ticker, date)
);

SELECT create_hypertable('feature_technical_indicators', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_feature_technical_ticker_date 
ON feature_technical_indicators(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_feature_technical_computed_at 
ON feature_technical_indicators(computed_at DESC);

-- Market context features (aggregate features)
CREATE TABLE IF NOT EXISTS feature_market_context (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    -- Market breadth
    market_sentiment VARCHAR(20),  -- 'bullish', 'bearish', 'neutral'
    sector_momentum DECIMAL(8, 6),  -- Average returns of sector
    sector_volatility DECIMAL(5, 2),  -- Volatility within sector
    -- Correlation features
    correlation_sp500 DECIMAL(4, 3),  -- Correlation with S&P 500
    correlation_sector DECIMAL(4, 3),  -- Correlation with sector
    correlation_tech DECIMAL(4, 3),  -- Correlation with tech index
    -- Market conditions
    market_volatility DECIMAL(5, 2),  -- VIX-like proxy
    market_trend VARCHAR(10),  -- 'uptrend', 'downtrend', 'sideways'
    market_trend_strength DECIMAL(4, 3),  -- 0-1 strength of trend
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    feature_version INT NOT NULL DEFAULT 1,
    UNIQUE(ticker, date)
);

SELECT create_hypertable('feature_market_context', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_feature_context_ticker_date 
ON feature_market_context(ticker, date DESC);

-- ============================================================================
-- 5. TARGET VARIABLE TABLES
-- ============================================================================

-- Daily target variable (next-day price movement)
CREATE TABLE IF NOT EXISTS target_daily (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    target_return DECIMAL(8, 6) NOT NULL,  -- (next_close - close) / close
    target_binary INT,  -- 1 if positive return, 0 if negative/flat
    target_direction VARCHAR(10),  -- 'up', 'down', 'flat'
    confidence DECIMAL(3, 2),  -- Confidence in the target
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

SELECT create_hypertable('target_daily', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_target_daily_ticker_date 
ON target_daily(ticker, date DESC);

-- ============================================================================
-- 6. MODEL METADATA TABLES
-- ============================================================================

-- Training data snapshots (for reproducibility)
CREATE TABLE IF NOT EXISTS training_dataset_metadata (
    id BIGSERIAL PRIMARY KEY,
    dataset_id VARCHAR(100) NOT NULL UNIQUE,
    dataset_name VARCHAR(255) NOT NULL,
    version INT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    tickers TEXT[] NOT NULL,  -- Array of tickers used
    total_records INT NOT NULL,
    train_records INT NOT NULL,
    validation_records INT NOT NULL,
    test_records INT NOT NULL,
    feature_count INT NOT NULL,
    features_used TEXT[],  -- Array of feature names
    preprocessing_steps JSONB,
    data_split_method VARCHAR(50),  -- 'temporal', 'random', 'stratified'
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    validation_score DECIMAL(5, 4),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_dataset_name_version 
ON training_dataset_metadata(dataset_name, version DESC);

-- Feature importance tracking
CREATE TABLE IF NOT EXISTS feature_importance (
    id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    importance_score DECIMAL(8, 6) NOT NULL,
    importance_type VARCHAR(50),  -- 'shap', 'permutation', 'gain'
    rank INT,
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_version, feature_name, importance_type)
);

CREATE INDEX IF NOT EXISTS idx_feature_importance_model_version 
ON feature_importance(model_version);

-- ============================================================================
-- 7. DATA RETENTION POLICIES
-- ============================================================================

-- Drop old raw data after processing (keep 2 years)
-- SELECT set_integer_config('timescaledb.compress_orderby', 'date DESC');
-- SELECT add_compression_policy('raw_market_data', INTERVAL '730 days');

-- Archive processed data after 5 years (move to S3)
-- SELECT add_retention_policy('processed_market_data', INTERVAL '1825 days');

-- ============================================================================
-- 8. HELPER MATERIALIZED VIEWS
-- ============================================================================

-- Latest market data per ticker
CREATE MATERIALIZED VIEW IF NOT EXISTS latest_market_data_mv AS
SELECT DISTINCT ON (ticker)
    ticker,
    date,
    close,
    volume,
    ingested_at
FROM raw_market_data
ORDER BY ticker, date DESC;

CREATE INDEX IF NOT EXISTS idx_latest_market_data_ticker 
ON latest_market_data_mv(ticker);

-- ============================================================================
-- 9. AUDIT & TRACKING TABLES
-- ============================================================================

-- Track all data modifications for compliance
CREATE TABLE IF NOT EXISTS data_lineage (
    id BIGSERIAL PRIMARY KEY,
    source_table VARCHAR(100) NOT NULL,
    target_table VARCHAR(100) NOT NULL,
    transformation_name VARCHAR(255) NOT NULL,
    successful BOOLEAN NOT NULL,
    records_input INT,
    records_output INT,
    execution_time_ms INT,
    executed_by VARCHAR(100),
    executed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    lineage_details JSONB
);

CREATE INDEX IF NOT EXISTS idx_lineage_source_target 
ON data_lineage(source_table, target_table);
CREATE INDEX IF NOT EXISTS idx_lineage_executed_at 
ON data_lineage(executed_at DESC);

-- ============================================================================
-- 10. INITIALIZATION DATA
-- ============================================================================

-- Insert monitoring tickers
INSERT INTO raw_market_data (ticker, date, open, high, low, close, volume, source)
VALUES 
    ('AAPL', CURRENT_DATE - INTERVAL '1 day', 150.0, 152.0, 149.5, 151.5, 50000000, 'yfinance'),
    ('GOOGL', CURRENT_DATE - INTERVAL '1 day', 140.0, 142.0, 139.5, 141.5, 40000000, 'yfinance'),
    ('MSFT', CURRENT_DATE - INTERVAL '1 day', 300.0, 305.0, 299.5, 304.0, 30000000, 'yfinance')
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 11. MAINTENANCE FUNCTIONS
-- ============================================================================

-- Function to validate data quality
CREATE OR REPLACE FUNCTION validate_market_data()
RETURNS TABLE (
    ticker VARCHAR,
    validation_date DATE,
    has_missing BOOLEAN,
    has_outliers BOOLEAN,
    quality_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        rmd.ticker,
        rmd.date,
        (open IS NULL OR close IS NULL OR volume IS NULL) as has_missing,
        (rmd.close > rmd.high OR rmd.close < rmd.low) as has_outliers,
        COALESCE(dqm.quality_score, 0.5) as quality_score
    FROM raw_market_data rmd
    LEFT JOIN data_quality_metrics dqm ON rmd.ticker = dqm.ticker AND rmd.date = dqm.date
    WHERE rmd.date = CURRENT_DATE - INTERVAL '1 day'
    ORDER BY ticker;
END;
$$ LANGUAGE plpgsql;

-- Function to compute technical indicator (RSI example)
CREATE OR REPLACE FUNCTION compute_rsi(
    p_ticker VARCHAR,
    p_date DATE,
    p_period INT DEFAULT 14
)
RETURNS DECIMAL AS $$
DECLARE
    v_avg_gain DECIMAL;
    v_avg_loss DECIMAL;
    v_rs DECIMAL;
    v_rsi DECIMAL;
BEGIN
    -- Simplified RSI calculation
    -- In production, use vectorized computation in Python
    RETURN (50.0);  -- Placeholder
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANT PERMISSIONS (Adjust based on your environment)
-- ============================================================================

-- Application user (read/write to main tables)
-- CREATE USER app_user WITH PASSWORD 'secure_password';
-- GRANT CONNECT ON DATABASE stock_prediction TO app_user;
-- GRANT USAGE ON SCHEMA public TO app_user;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO app_user;

-- Analytics user (read-only)
-- CREATE USER analytics_user WITH PASSWORD 'secure_password';
-- GRANT CONNECT ON DATABASE stock_prediction TO analytics_user;
-- GRANT USAGE ON SCHEMA public TO analytics_user;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO analytics_user;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
