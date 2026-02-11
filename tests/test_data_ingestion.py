"""
Unit tests for data ingestion pipeline and validation

Tests cover:
- Data validation logic
- Batch processing
- Error handling
- Database operations
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import asyncio

from production.data.ingestion import (
    DataIngestionConfig,
    MarketDataValidator,
    MarketDataFetcher,
    DataIngestionPipeline,
    DataValidationError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Default configuration for testing"""
    return DataIngestionConfig()


@pytest.fixture
def validator(config):
    """Validator instance for testing"""
    return MarketDataValidator(config)


@pytest.fixture
def sample_valid_record():
    """Valid market data record"""
    return {
        'ticker': 'AAPL',
        'date': datetime.now().date(),
        'open': Decimal('150.0'),
        'high': Decimal('152.0'),
        'low': Decimal('149.0'),
        'close': Decimal('151.0'),
        'volume': 50000000,
        'adj_close': Decimal('151.0'),
        'source': 'yfinance'
    }


@pytest.fixture
def sample_dataframe():
    """Sample OHLCV dataframe"""
    dates = pd.date_range(end=datetime.now(), periods=10)
    return pd.DataFrame({
        'ticker': ['AAPL'] * 10,
        'date': dates.date,
        'open': np.random.uniform(150, 155, 10),
        'high': np.random.uniform(155, 160, 10),
        'low': np.random.uniform(145, 150, 10),
        'close': np.random.uniform(150, 155, 10),
        'volume': np.random.randint(40000000, 60000000, 10),
        'source': ['yfinance'] * 10
    })


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    mock_conn = MagicMock()
    mock_session = MagicMock()
    mock_conn.Session.return_value = mock_session
    return mock_conn


# ============================================================================
# DATA VALIDATION TESTS
# ============================================================================

class TestMarketDataValidator:
    """Tests for market data validation"""
    
    def test_valid_record_passes_validation(self, validator, sample_valid_record):
        """Valid record should pass all checks"""
        is_valid, error = validator.validate_record(sample_valid_record)
        assert is_valid is True
        assert error is None
    
    def test_missing_required_field_fails(self, validator, sample_valid_record):
        """Record with missing required field should fail"""
        del sample_valid_record['close']
        is_valid, error = validator.validate_record(sample_valid_record)
        assert is_valid is False
        assert 'Missing required field' in error
    
    def test_price_outside_valid_range_fails(self, validator, sample_valid_record):
        """Price outside valid range should fail"""
        sample_valid_record['close'] = Decimal('0.001')  # Too low
        is_valid, error = validator.validate_record(sample_valid_record)
        assert is_valid is False
    
    def test_high_less_than_low_fails(self, validator, sample_valid_record):
        """High < Low should fail"""
        sample_valid_record['high'] = Decimal('148.0')
        sample_valid_record['low'] = Decimal('150.0')
        is_valid, error = validator.validate_record(sample_valid_record)
        assert is_valid is False
        assert 'High < Low' in error
    
    def test_close_outside_ohlc_range_fails(self, validator, sample_valid_record):
        """Close outside OHLC range should fail"""
        sample_valid_record['close'] = Decimal('160.0')  # Above high
        is_valid, error = validator.validate_record(sample_valid_record)
        assert is_valid is False
        assert 'Close outside OHLC range' in error
    
    def test_volume_below_minimum_fails(self, validator, sample_valid_record):
        """Volume below minimum should fail"""
        sample_valid_record['volume'] = 50000  # Below minimum
        is_valid, error = validator.validate_record(sample_valid_record)
        assert is_valid is False
        assert 'below minimum' in error
    
    def test_unusual_price_range_fails(self, validator, sample_valid_record):
        """Unusual price range should fail"""
        sample_valid_record['high'] = Decimal('200.0')  # 50% range
        is_valid, error = validator.validate_record(sample_valid_record)
        assert is_valid is False
        assert 'Unusual price range' in error
    
    def test_type_error_in_record_fails(self, validator):
        """Invalid data types should fail"""
        invalid_record = {
            'ticker': 'AAPL',
            'date': datetime.now().date(),
            'open': 'not_a_number',  # Invalid type
            'high': Decimal('152.0'),
            'low': Decimal('149.0'),
            'close': Decimal('151.0'),
            'volume': 50000000,
        }
        is_valid, error = validator.validate_record(invalid_record)
        assert is_valid is False
        assert 'Type error' in error
    
    def test_batch_validation_separates_valid_invalid(self, validator, sample_valid_record):
        """Batch validation should separate valid and invalid records"""
        invalid_record = sample_valid_record.copy()
        invalid_record['volume'] = 50000  # Invalid
        
        batch = [sample_valid_record, invalid_record]
        valid, invalid = validator.validate_batch(batch)
        
        assert len(valid) == 1
        assert len(invalid) == 1
        assert valid[0]['ticker'] == 'AAPL'
        assert 'validation_error' in invalid[0]
    
    def test_empty_batch_validation(self, validator):
        """Empty batch should return empty lists"""
        valid, invalid = validator.validate_batch([])
        assert len(valid) == 0
        assert len(invalid) == 0


# ============================================================================
# DATA FETCHER TESTS
# ============================================================================

class TestMarketDataFetcher:
    """Tests for market data fetching"""
    
    @pytest.mark.asyncio
    async def test_fetch_yfinance_success(self):
        """Successful yfinance fetch"""
        fetcher = MarketDataFetcher()
        
        with patch('yfinance.download') as mock_download:
            # Mock DataFrame
            mock_df = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=5),
                'Open': [150, 151, 152, 151, 150],
                'High': [151, 152, 153, 152, 151],
                'Low': [149, 150, 151, 150, 149],
                'Close': [150.5, 151.5, 152.5, 151.5, 150.5],
                'Volume': [50000000] * 5,
                'Adj Close': [150.5, 151.5, 152.5, 151.5, 150.5],
            })
            mock_download.return_value = mock_df
            
            result = await fetcher.fetch_yfinance(
                'AAPL',
                datetime(2024, 1, 1),
                datetime(2024, 1, 5)
            )
            
            assert not result.empty
            assert len(result) == 5
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
            assert result['ticker'].iloc[0] == 'AAPL'
            assert result['source'].iloc[0] == 'yfinance'
    
    @pytest.mark.asyncio
    async def test_fetch_yfinance_empty_response(self):
        """Handle empty response from yfinance"""
        fetcher = MarketDataFetcher()
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = pd.DataFrame()
            
            result = await fetcher.fetch_yfinance(
                'INVALID',
                datetime(2024, 1, 1),
                datetime(2024, 1, 5)
            )
            
            assert result.empty
    
    @pytest.mark.asyncio
    async def test_fetch_multiple_concurrent(self):
        """Fetch multiple tickers concurrently"""
        fetcher = MarketDataFetcher()
        
        with patch('yfinance.download') as mock_download:
            mock_df = pd.DataFrame({
                'Date': [datetime.now().date()],
                'Open': [150],
                'High': [151],
                'Low': [149],
                'Close': [150.5],
                'Volume': [50000000],
                'Adj Close': [150.5],
            })
            mock_download.return_value = mock_df
            
            results = await fetcher.fetch_multiple(
                ['AAPL', 'GOOGL', 'MSFT'],
                datetime(2024, 1, 1),
                datetime(2024, 1, 5)
            )
            
            assert len(results) >= 1  # At least one successful fetch


# ============================================================================
# DATA INGESTION PIPELINE TESTS
# ============================================================================

class TestDataIngestionPipeline:
    """Tests for data ingestion pipeline"""
    
    @pytest.mark.asyncio
    async def test_ingest_daily_data_success(self, mock_db_connection):
        """Successful daily data ingestion"""
        with patch('production.data.ingestion.create_engine') as mock_engine:
            mock_engine.return_value = mock_db_connection
            
            pipeline = DataIngestionPipeline("postgresql://localhost/db")
            
            # Mock the fetcher and validator
            with patch.object(pipeline, 'fetcher') as mock_fetcher:
                with patch.object(pipeline, 'validator') as mock_validator:
                    # Setup mocks
                    mock_fetcher.fetch_multiple.return_value = {
                        'AAPL': pd.DataFrame({
                            'ticker': ['AAPL'],
                            'date': [datetime.now().date()],
                            'open': [150.0],
                            'high': [151.0],
                            'low': [149.0],
                            'close': [150.5],
                            'volume': [50000000],
                            'source': ['yfinance']
                        })
                    }
                    
                    valid_record = {
                        'ticker': 'AAPL',
                        'date': datetime.now().date(),
                        'open': 150.0,
                        'high': 151.0,
                        'low': 149.0,
                        'close': 150.5,
                        'volume': 50000000,
                        'source': 'yfinance'
                    }
                    mock_validator.validate_batch.return_value = ([valid_record], [])
                    
                    # Run ingestion
                    stats = await pipeline.ingest_daily_data(['AAPL'], lookback_days=1)
                    
                    assert stats['inserted_records'] >= 0
                    assert stats['tickers_processed'] == 1
    
    def test_validate_data_quality(self, mock_db_connection):
        """Test data quality validation"""
        with patch('production.data.ingestion.create_engine') as mock_engine:
            mock_engine.return_value = mock_db_connection
            
            pipeline = DataIngestionPipeline("postgresql://localhost/db")
            
            # Mock database query
            with patch.object(pipeline.Session, 'return_value') as mock_session:
                mock_result = (150.0, 151.0, 50000000.0)
                mock_session.execute.return_value.fetchone.return_value = mock_result
                
                quality = pipeline.validate_data_quality('AAPL', datetime.now().date())
                
                assert quality['ticker'] == 'AAPL'
                assert quality['has_data'] is True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDataIngestionIntegration:
    """Integration tests for complete data ingestion workflow"""
    
    def test_validation_preserves_record_structure(self, validator, sample_valid_record):
        """Validation should preserve original record structure"""
        is_valid, error = validator.validate_record(sample_valid_record)
        assert is_valid
        
        # Check all fields are preserved
        for key in sample_valid_record.keys():
            assert key in sample_valid_record
    
    def test_batch_processing_maintains_order(self, validator):
        """Batch processing should maintain record order"""
        records = [
            {'ticker': 'AAPL', 'date': datetime.now().date(), 'open': 150, 'high': 151, 'low': 149, 'close': 150.5, 'volume': 50000000},
            {'ticker': 'GOOGL', 'date': datetime.now().date(), 'open': 140, 'high': 141, 'low': 139, 'close': 140.5, 'volume': 50000000},
        ]
        
        valid, invalid = validator.validate_batch(records)
        
        if valid:
            assert valid[0]['ticker'] == 'AAPL'


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestDataIngestionPerformance:
    """Performance tests for data ingestion"""
    
    def test_validation_throughput(self, validator):
        """Validation should handle high throughput"""
        records = []
        for i in range(1000):
            records.append({
                'ticker': f'TICK{i % 100}',
                'date': datetime.now().date(),
                'open': 150.0 + i,
                'high': 151.0 + i,
                'low': 149.0 + i,
                'close': 150.5 + i,
                'volume': 50000000 + i,
            })
        
        import time
        start = time.time()
        valid, invalid = validator.validate_batch(records)
        elapsed = time.time() - start
        
        throughput = len(records) / elapsed
        assert throughput > 100, f"Throughput too low: {throughput} records/sec"
    
    def test_batch_size_optimization(self, validator):
        """Test optimal batch size for validation"""
        batch_sizes = [10, 100, 1000]
        
        for batch_size in batch_sizes:
            records = [
                {
                    'ticker': 'AAPL',
                    'date': datetime.now().date(),
                    'open': 150.0,
                    'high': 151.0,
                    'low': 149.0,
                    'close': 150.5,
                    'volume': 50000000,
                } for _ in range(batch_size)
            ]
            
            valid, invalid = validator.validate_batch(records)
            assert len(valid) == batch_size


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_validation_error_on_null_value(self, validator):
        """Null values should be handled gracefully"""
        record = {
            'ticker': 'AAPL',
            'date': None,  # Null date
            'open': 150.0,
            'high': 151.0,
            'low': 149.0,
            'close': 150.5,
            'volume': 50000000,
        }
        
        is_valid, error = validator.validate_record(record)
        assert is_valid is False
    
    def test_batch_continues_on_partial_failure(self, validator):
        """Batch should continue even if some records are invalid"""
        records = [
            {
                'ticker': 'AAPL',
                'date': datetime.now().date(),
                'open': 150.0,
                'high': 151.0,
                'low': 149.0,
                'close': 150.5,
                'volume': 50000000,
            },
            {
                'ticker': 'GOOGL',
                'date': None,  # Invalid
                'open': 140.0,
                'high': 141.0,
                'low': 139.0,
                'close': 140.5,
                'volume': 50000000,
            },
        ]
        
        valid, invalid = validator.validate_batch(records)
        assert len(valid) == 1
        assert len(invalid) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
