# ADR 008: Data Storage Strategy - PostgreSQL vs MongoDB vs Data Lake

**Status:** ACCEPTED  
**Date:** February 2026  
**ADR Number:** 008  

---

## Decision

**SELECTED: PostgreSQL (primary) + S3 Data Lake (secondary)**

### Pros
- ✅ PostgreSQL: ACID guarantees, excellent performance, JSON support
- ✅ S3: Unlimited scale, cost-effective for archives
- ✅ Time-series extension: pg_partman for efficient time-series
- ✅ Hybrid approach: Hot data in DB, cold data in S3

### Implementation

```sql
-- PostgreSQL schema

CREATE TABLE stock_prices (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date)
);

-- Time-series partitioning
SELECT create_hypertable('stock_prices', 'date');

-- Indexes for common queries
CREATE INDEX idx_ticker_date ON stock_prices (ticker, date DESC);
CREATE INDEX idx_created ON stock_prices (created_at);

-- Archive to S3 after 2 years
CREATE POLICY archive_old_data AS (
    DELETE FROM stock_prices
    WHERE date < CURRENT_DATE - INTERVAL '2 years'
    RETURNING *;
);
```

```python
# Archive process

import s3fs
import pandas as pd

def archive_old_data():
    """Archive data older than 2 years to S3"""
    cutoff_date = (datetime.now() - timedelta(days=730)).date()
    
    # Query old data
    sql = f"SELECT * FROM stock_prices WHERE date < '{cutoff_date}'"
    df = pd.read_sql(sql, con=db_connection)
    
    # Save to S3
    s3 = s3fs.S3FileSystem()
    year = cutoff_date.year
    s3_path = f"s3://data-lake/stock-prices/year={year}/data.parquet"
    df.to_parquet(s3_path)
    
    # Delete from DB
    delete_sql = f"DELETE FROM stock_prices WHERE date < '{cutoff_date}'"
    db_connection.execute(delete_sql)
```

---

**Status:** ✅ ACCEPTED
