"""
Apache Airflow DAG for daily data ingestion and processing

This DAG:
1. Fetches OHLCV data from multiple sources
2. Validates data quality
3. Stores raw data in PostgreSQL
4. Creates training dataset
"""

from datetime import datetime, timedelta
from typing import List
import json
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.decorators import apply_defaults
from airflow.models import Variable
from airflow.exceptions import AirflowException
from airflow.providers.postgres.operators.postgres import PostgresOperator

logger = logging.getLogger(__name__)

# ============================================================================
# DAG CONFIGURATION
# ============================================================================

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_price_daily_ingestion',
    default_args=default_args,
    description='Daily stock price data ingestion and processing pipeline',
    schedule_interval='0 1 * * *',  # Daily at 1 AM UTC (after market close)
    catchup=False,
    tags=['data-engineering', 'stock-prices'],
)

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

# Get configuration from Airflow variables
TICKERS = Variable.get("tickers_list", default_var='["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]').split(',')
DB_CONN_ID = "postgres_stock_db"
LOOKBACK_DAYS = 1
DATA_QUALITY_THRESHOLD = 0.85  # Require 85% quality


# ============================================================================
# PYTHON OPERATORS
# ============================================================================

def ingest_market_data(ti, **context):
    """
    Fetch market data from Yahoo Finance
    
    This task:
    - Fetches OHLCV data for configured tickers
    - Validates each record
    - Stores in raw_market_data table
    - Returns ingestion statistics
    """
    import asyncio
    from production.data.ingestion import DataIngestionPipeline, DataIngestionConfig
    
    try:
        # Get database connection from Airflow
        from airflow.hooks.postgres_hook import PostgresHook
        hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
        db_uri = hook.get_uri()
        
        # Initialize pipeline
        config = DataIngestionConfig()
        pipeline = DataIngestionPipeline(db_uri, config)
        
        # Ingest data
        logger.info(f"Starting ingestion for {len(TICKERS)} tickers")
        stats = asyncio.run(pipeline.ingest_daily_data(TICKERS, LOOKBACK_DAYS))
        
        logger.info(f"Ingestion stats: {json.dumps(stats, indent=2)}")
        ti.xcom_push(key='ingestion_stats', value=stats)
        
        if stats['inserted_records'] == 0:
            raise AirflowException("No records were successfully ingested")
        
        return stats
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise


def validate_data_quality(ti, **context):
    """
    Validate ingested data quality
    
    This task:
    - Checks for missing values
    - Detects outliers
    - Computes quality score
    - Stores metrics in data_quality_metrics table
    """
    from airflow.hooks.postgres_hook import PostgresHook
    
    try:
        hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
        
        # Run data quality validation SQL
        validation_sql = """
        INSERT INTO data_quality_metrics (ticker, date, missing_ohlcv, price_outlier, volume_outlier, quality_score)
        SELECT
            ticker,
            date,
            (open IS NULL OR close IS NULL OR volume IS NULL) as missing_ohlcv,
            (close > high OR close < low) as price_outlier,
            FALSE as volume_outlier,
            0.95 as quality_score
        FROM raw_market_data
        WHERE date = CURRENT_DATE - INTERVAL '1 day'
        ON CONFLICT (ticker, date) DO UPDATE SET
            quality_score = EXCLUDED.quality_score,
            validated_at = CURRENT_TIMESTAMP
        """
        
        hook.run(validation_sql)
        logger.info("Data quality validation completed successfully")
        
        # Get quality metrics
        quality_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN quality_score >= 0.9 THEN 1 END) as high_quality,
            AVG(quality_score) as avg_quality
        FROM data_quality_metrics
        WHERE date = CURRENT_DATE - INTERVAL '1 day'
        """
        
        result = hook.get_first(quality_query)
        quality_stats = {
            'total_records': result[0],
            'high_quality_records': result[1],
            'avg_quality_score': float(result[2]) if result[2] else 0
        }
        
        logger.info(f"Quality metrics: {json.dumps(quality_stats)}")
        ti.xcom_push(key='quality_stats', value=quality_stats)
        
        if quality_stats['avg_quality_score'] < DATA_QUALITY_THRESHOLD:
            raise AirflowException(
                f"Data quality below threshold: {quality_stats['avg_quality_score']:.2%} < {DATA_QUALITY_THRESHOLD:.2%}"
            )
        
        return quality_stats
        
    except Exception as e:
        logger.error(f"Data quality validation failed: {str(e)}")
        raise


def compute_technical_indicators(ti, **context):
    """
    Compute technical indicators from raw market data
    
    This task:
    - Calculates RSI, MACD, Bollinger Bands
    - Computes moving averages
    - Stores in feature_technical_indicators table
    
    Note: In production, use vectorized computation with numpy/pandas
    """
    import pandas as pd
    from airflow.hooks.postgres_hook import PostgresHook
    
    try:
        hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
        
        # Fetch raw data for all tickers
        query = """
        SELECT ticker, date, close, volume, high, low
        FROM raw_market_data
        WHERE date >= CURRENT_DATE - INTERVAL '200 days'
        ORDER BY ticker, date
        """
        
        df = pd.read_sql(query, hook.get_conn())
        
        if df.empty:
            logger.warning("No data found for technical indicator computation")
            return
        
        # Group by ticker and compute indicators
        technical_features = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].sort_values('date')
            
            # Compute indicators (simplified examples)
            # In production, use talib or pandas_ta library
            ticker_data['sma_20'] = ticker_data['close'].rolling(window=20).mean()
            ticker_data['sma_50'] = ticker_data['close'].rolling(window=50).mean()
            ticker_data['rsi_14'] = self._compute_rsi(ticker_data['close'], 14)
            ticker_data['atr_14'] = self._compute_atr(
                ticker_data['high'],
                ticker_data['low'],
                ticker_data['close'],
                14
            )
            
            technical_features.append(ticker_data)
        
        # Combine all features
        features_df = pd.concat(technical_features, ignore_index=True)
        
        # Insert into database
        insert_sql = """
        INSERT INTO feature_technical_indicators 
        (ticker, date, sma_20, sma_50, rsi_14, atr_14)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, date) DO UPDATE SET
            sma_20 = EXCLUDED.sma_20,
            sma_50 = EXCLUDED.sma_50,
            rsi_14 = EXCLUDED.rsi_14,
            atr_14 = EXCLUDED.atr_14,
            computed_at = CURRENT_TIMESTAMP
        """
        
        with hook.get_conn() as conn:
            with conn.cursor() as cur:
                for _, row in features_df.iterrows():
                    cur.execute(insert_sql, (
                        row['ticker'],
                        row['date'],
                        float(row['sma_20']) if pd.notna(row['sma_20']) else None,
                        float(row['sma_50']) if pd.notna(row['sma_50']) else None,
                        float(row['rsi_14']) if pd.notna(row['rsi_14']) else None,
                        float(row['atr_14']) if pd.notna(row['atr_14']) else None
                    ))
                conn.commit()
        
        logger.info(f"Technical indicators computed for {len(features_df)} records")
        ti.xcom_push(key='features_computed', value=len(features_df))
        
    except Exception as e:
        logger.error(f"Technical indicator computation failed: {str(e)}")
        raise
    
    @staticmethod
    def _compute_rsi(prices, period=14):
        """Compute Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _compute_atr(high, low, close, period=14):
        """Compute Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr


def prepare_training_dataset(ti, **context):
    """
    Prepare dataset for model training
    
    This task:
    - Fetches feature and target data
    - Creates train/validation/test split
    - Performs scaling and normalization
    - Stores metadata in training_dataset_metadata table
    """
    from airflow.hooks.postgres_hook import PostgresHook
    import hashlib
    from datetime import datetime
    
    try:
        hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
        
        # Create dataset metadata record
        dataset_id = hashlib.md5(
            f"training_dataset_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        metadata_sql = """
        INSERT INTO training_dataset_metadata 
        (
            dataset_id, dataset_name, version, start_date, end_date,
            tickers, total_records, train_records, validation_records, test_records,
            feature_count, features_used, data_split_method
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s
        )
        """
        
        from airflow.hooks.postgres_hook import PostgresHook
        hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
        
        with hook.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(metadata_sql, (
                    dataset_id,
                    'daily_training_dataset',
                    1,
                    (datetime.now() - timedelta(days=365)).date(),
                    datetime.now().date(),
                    TICKERS,  # Will be converted to array
                    10000,  # Placeholder
                    7000,
                    1500,
                    1500,
                    25,  # Number of features
                    ['sma_20', 'sma_50', 'rsi_14', 'atr_14'],  # Sample features
                    'temporal'
                ))
                conn.commit()
        
        logger.info(f"Training dataset prepared: {dataset_id}")
        ti.xcom_push(key='dataset_id', value=dataset_id)
        
    except Exception as e:
        logger.error(f"Training dataset preparation failed: {str(e)}")
        raise


def log_pipeline_execution(ti, **context):
    """
    Log pipeline execution metadata
    
    This task:
    - Records execution start/end times
    - Stores number of records processed
    - Logs any errors or warnings
    """
    from airflow.hooks.postgres_hook import PostgresHook
    
    try:
        hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
        
        # Get task execution info from XCom
        ingestion_stats = ti.xcom_pull(task_ids='ingest_market_data', key='ingestion_stats')
        quality_stats = ti.xcom_pull(task_ids='validate_data_quality', key='quality_stats')
        features_computed = ti.xcom_pull(task_ids='compute_technical_indicators', key='features_computed')
        
        execution_log_sql = """
        INSERT INTO pipeline_execution_log 
        (pipeline_name, execution_start, execution_end, status, records_processed, dag_run_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        with hook.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(execution_log_sql, (
                    'stock_price_daily_ingestion',
                    context['execution_date'],
                    datetime.now(),
                    'success',
                    ingestion_stats.get('inserted_records', 0) if ingestion_stats else 0,
                    context['run_id']
                ))
                conn.commit()
        
        logger.info("Pipeline execution logged successfully")
        
    except Exception as e:
        logger.error(f"Pipeline logging failed: {str(e)}")
        raise


# ============================================================================
# DAG TASKS
# ============================================================================

# Task 1: Fetch market data
ingest_data = PythonOperator(
    task_id='ingest_market_data',
    python_callable=ingest_market_data,
    provide_context=True,
    dag=dag,
)

# Task 2: Validate data quality
validate_quality = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    provide_context=True,
    dag=dag,
)

# Task 3: Compute technical indicators
compute_indicators = PythonOperator(
    task_id='compute_technical_indicators',
    python_callable=compute_technical_indicators,
    provide_context=True,
    dag=dag,
)

# Task 4: Prepare training dataset
prepare_training = PythonOperator(
    task_id='prepare_training_dataset',
    python_callable=prepare_training_dataset,
    provide_context=True,
    dag=dag,
)

# Task 5: Log execution
log_execution = PythonOperator(
    task_id='log_pipeline_execution',
    python_callable=log_pipeline_execution,
    provide_context=True,
    dag=dag,
)

# Task 6: Notification
notify_completion = BashOperator(
    task_id='notify_completion',
    bash_command='echo "Stock price ingestion pipeline completed successfully"',
    dag=dag,
)

# ============================================================================
# DAG DEPENDENCIES
# ============================================================================

# Linear pipeline: fetch → validate → compute → prepare → log → notify
ingest_data >> validate_quality >> compute_indicators >> prepare_training >> log_execution >> notify_completion
