"""
Data Ingestion Module - Multi-source stock market data collection

Responsible for:
- Fetching OHLCV data from multiple sources
- Validating ingested data
- Storing raw data in PostgreSQL
- Handling retries and error recovery
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import json

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import yfinance as yf
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    pass


class DataIngestionConfig:
    """Configuration for data ingestion"""
    
    def __init__(self):
        self.batch_size = 100
        self.max_retries = 3
        self.retry_backoff = 2
        self.validation_rules = {
            'min_price': Decimal('0.01'),
            'max_price': Decimal('100000.00'),
            'min_volume': 100000,
            'max_volume_spike': 10.0,  # 10x normal
        }
        self.outlier_threshold = 3  # Standard deviations


class MarketDataValidator:
    """Validates market data for quality issues"""
    
    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()
    
    def validate_record(self, record: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate a single market data record
        
        Args:
            record: OHLCV data record
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check required fields
            required_fields = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if field not in record or record[field] is None:
                    return False, f"Missing required field: {field}"
            
            # Parse values
            close = Decimal(str(record['close']))
            high = Decimal(str(record['high']))
            low = Decimal(str(record['low']))
            open_price = Decimal(str(record['open']))
            volume = int(record['volume'])
            
            # Validation rules
            if not (self.config.validation_rules['min_price'] <= close <= 
                    self.config.validation_rules['max_price']):
                return False, f"Price {close} outside valid range"
            
            if high < low:
                return False, "High < Low"
            
            if close > high or close < low:
                return False, "Close outside OHLC range"
            
            if volume < self.config.validation_rules['min_volume']:
                return False, f"Volume {volume} below minimum"
            
            # Check for suspicious patterns
            price_range = (high - low) / low
            if price_range > 0.5:  # More than 50% range
                return False, f"Unusual price range: {price_range:.2%}"
            
            return True, None
            
        except (ValueError, TypeError, decimal.InvalidOperation) as e:
            return False, f"Type error: {str(e)}"
    
    def validate_batch(self, records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate a batch of records
        
        Args:
            records: List of market data records
            
        Returns:
            (valid_records, invalid_records)
        """
        valid = []
        invalid = []
        
        for record in records:
            is_valid, error = self.validate_record(record)
            if is_valid:
                valid.append(record)
            else:
                record['validation_error'] = error
                invalid.append(record)
        
        return valid, invalid


class MarketDataFetcher:
    """Fetches market data from various sources"""
    
    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, TimeoutError))
    )
    async def fetch_yfinance(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {ticker} from yfinance: {start_date} to {end_date}")
            
            data = await asyncio.to_thread(
                yf.download,
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data received for {ticker}")
                return pd.DataFrame()
            
            # Normalize columns
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            data['ticker'] = ticker
            data['source'] = 'yfinance'
            data = data.reset_index()
            data['date'] = pd.to_datetime(data['Date']).dt.date
            
            logger.info(f"Successfully fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}")
            raise
    
    async def fetch_multiple(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers concurrently
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        tasks = [
            self.fetch_yfinance(ticker, start_date, end_date)
            for ticker in tickers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {ticker}: {result}")
                continue
            data_dict[ticker] = result
        
        return data_dict


class DataIngestionPipeline:
    """Main data ingestion pipeline orchestrator"""
    
    def __init__(self, db_connection_string: str, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()
        self.engine = create_engine(db_connection_string)
        self.Session = sessionmaker(bind=self.engine)
        self.validator = MarketDataValidator(config)
        self.fetcher = MarketDataFetcher(config)
    
    async def ingest_daily_data(
        self,
        tickers: List[str],
        lookback_days: int = 1
    ) -> Dict[str, int]:
        """
        Ingest daily OHLCV data for specified tickers
        
        Args:
            tickers: List of ticker symbols
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with ingestion statistics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(f"Starting ingestion for {len(tickers)} tickers: {start_date} to {end_date}")
        
        # Fetch data from all sources
        fetched_data = await self.fetcher.fetch_multiple(
            tickers,
            start_date,
            end_date
        )
        
        # Process each ticker's data
        stats = {
            'total_records': 0,
            'inserted_records': 0,
            'validation_errors': 0,
            'tickers_processed': 0
        }
        
        session = self.Session()
        try:
            for ticker, df in fetched_data.items():
                if df.empty:
                    continue
                
                records = df.to_dict('records')
                valid_records, invalid_records = self.validator.validate_batch(records)
                
                stats['validation_errors'] += len(invalid_records)
                logger.warning(f"{ticker}: {len(invalid_records)} validation errors")
                
                # Insert valid records
                inserted = self._insert_batch(session, valid_records)
                stats['inserted_records'] += inserted
                stats['total_records'] += len(records)
                stats['tickers_processed'] += 1
            
            session.commit()
            logger.info(f"Ingestion completed: {stats}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Ingestion pipeline failed: {str(e)}")
            raise
        finally:
            session.close()
        
        return stats
    
    def _insert_batch(self, session, records: List[Dict]) -> int:
        """Insert batch of records into database"""
        inserted = 0
        for record in records:
            try:
                sql = text("""
                    INSERT INTO raw_market_data 
                    (ticker, date, open, high, low, close, volume, adj_close, source)
                    VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :adj_close, :source)
                    ON CONFLICT (ticker, date, source) DO NOTHING
                """)
                
                session.execute(sql, {
                    'ticker': record['ticker'],
                    'date': record['date'],
                    'open': record['open'],
                    'high': record['high'],
                    'low': record['low'],
                    'close': record['close'],
                    'volume': record['volume'],
                    'adj_close': record.get('adj_close'),
                    'source': record.get('source', 'unknown')
                })
                inserted += 1
                
            except IntegrityError:
                session.rollback()
                logger.debug(f"Duplicate record for {record['ticker']} on {record['date']}")
            except Exception as e:
                logger.error(f"Error inserting record: {str(e)}")
                session.rollback()
                continue
        
        return inserted
    
    def validate_data_quality(self, ticker: str, date: datetime) -> Dict:
        """
        Check data quality metrics for a specific date
        
        Args:
            ticker: Stock ticker
            date: Date to check
            
        Returns:
            Quality metrics dictionary
        """
        session = self.Session()
        try:
            sql = text("""
                SELECT 
                    COUNT(*) as record_count,
                    MIN(close) as min_close,
                    MAX(close) as max_close,
                    AVG(volume) as avg_volume
                FROM raw_market_data
                WHERE ticker = :ticker AND date = :date
            """)
            
            result = session.execute(sql, {'ticker': ticker, 'date': date}).fetchone()
            
            return {
                'ticker': ticker,
                'date': date,
                'has_data': result[0] > 0,
                'min_price': float(result[1]) if result[1] else None,
                'max_price': float(result[2]) if result[2] else None,
                'avg_volume': float(result[3]) if result[3] else None
            }
            
        finally:
            session.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage of data ingestion pipeline"""
    
    # Configuration
    db_connection = "postgresql://user:password@localhost/stock_prediction"
    config = DataIngestionConfig()
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline(db_connection, config)
    
    # Ingest daily data for key tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
    stats = await pipeline.ingest_daily_data(tickers, lookback_days=5)
    
    print(f"Ingestion Statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    import decimal
    import requests
    
    asyncio.run(main())
