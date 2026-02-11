"""
Temporal Cross-Validation Framework

Implements time-series specific cross-validation splits to prevent:
- Data leakage (future data in training)
- Look-ahead bias
- Information leakage from validation into training

Prevents common ML mistakes in time-series modeling.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class SplitStrategy(Enum):
    """Cross-validation split strategies for time-series data"""
    TEMPORAL = "temporal"  # Walk-forward validation
    ANCHORED = "anchored"  # Growing training window
    SLIDING = "sliding"  # Fixed-size sliding window


@dataclass
class CVFold:
    """Represents a single cross-validation fold"""
    
    fold_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None
    
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    
    def __str__(self):
        return (
            f"Fold {self.fold_id}: "
            f"Train [{self.train_start.date()} - {self.train_end.date()}] "
            f"({self.train_size} samples) -> "
            f"Val [{self.val_start.date()} - {self.val_end.date()}] "
            f"({self.val_size} samples)"
        )


class TemporalCrossValidator:
    """Temporal cross-validation splitter for time-series data"""
    
    def __init__(
        self,
        n_splits: int = 5,
        train_window_days: int = 252,  # ~1 year of trading days
        val_window_days: int = 21,  # ~1 month
        test_window_days: Optional[int] = None,
        gap_days: int = 1,  # Gap between train and validation
        strategy: SplitStrategy = SplitStrategy.TEMPORAL,
        min_train_size: int = 100,
    ):
        """
        Initialize temporal cross-validator
        
        Args:
            n_splits: Number of CV folds to create
            train_window_days: Size of training window in days
            val_window_days: Size of validation window in days
            test_window_days: Size of test window (if None, no test set)
            gap_days: Gap between train and validation (prevents leakage)
            strategy: Split strategy to use
            min_train_size: Minimum samples required in training set
        """
        self.n_splits = n_splits
        self.train_window_days = train_window_days
        self.val_window_days = val_window_days
        self.test_window_days = test_window_days or 0
        self.gap_days = gap_days
        self.strategy = strategy
        self.min_train_size = min_train_size
        
        self.folds: List[CVFold] = []
    
    def split(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        ticker_column: str = 'ticker'
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Generate temporal cross-validation splits
        
        Args:
            data: DataFrame with time-series data
            date_column: Name of date column
            ticker_column: Name of ticker column (for grouping)
            
        Returns:
            List of (train, val, test) DataFrame tuples for each fold
        """
        if not self._validate_data(data, date_column):
            raise ValueError("Invalid data format for temporal CV")
        
        # Convert date column to datetime if necessary
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Get date range
        date_col = pd.to_datetime(data[date_column])
        start_date = date_col.min()
        end_date = date_col.max()
        
        # Calculate split points
        self.folds = self._generate_folds(start_date, end_date)
        
        if not self.folds:
            raise ValueError("Could not generate CV folds with given parameters")
        
        # Create splits
        splits = []
        for fold in self.folds:
            train_indices = (
                (data[date_column] >= fold.train_start) &
                (data[date_column] <= fold.train_end)
            )
            
            # Add gap to prevent leakage
            val_start = fold.val_start + timedelta(days=self.gap_days)
            
            val_indices = (
                (data[date_column] >= val_start) &
                (data[date_column] <= fold.val_end)
            )
            
            train_df = data[train_indices].copy()
            val_df = data[val_indices].copy()
            
            # Optional test set
            if fold.test_start and fold.test_end:
                test_start = fold.test_end + timedelta(days=self.gap_days)
                test_indices = (
                    (data[date_column] >= test_start) &
                    (data[date_column] <= fold.test_end)
                )
                test_df = data[test_indices].copy()
            else:
                test_df = pd.DataFrame()
            
            # Update fold sizes
            fold.train_size = len(train_df)
            fold.val_size = len(val_df)
            fold.test_size = len(test_df)
            
            # Validate fold
            if fold.train_size < self.min_train_size:
                logger.warning(
                    f"Fold {fold.fold_id}: Training set too small "
                    f"({fold.train_size} < {self.min_train_size})"
                )
                continue
            
            splits.append((train_df, val_df, test_df))
            logger.info(str(fold))
        
        return splits
    
    def _generate_folds(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[CVFold]:
        """Generate CV fold boundaries"""
        folds = []
        
        total_days = (end_date - start_date).days
        total_window = self.train_window_days + self.val_window_days + self.test_window_days
        
        if total_days < total_window:
            logger.warning(
                f"Data range ({total_days}d) < required window ({total_window}d)"
            )
            self.n_splits = 1
        
        # Calculate step size for walk-forward
        if self.strategy == SplitStrategy.TEMPORAL:
            step_days = max(1, self.val_window_days)  # Walk forward by validation window
        elif self.strategy == SplitStrategy.ANCHORED:
            step_days = self.val_window_days  # Growing window
        else:  # SLIDING
            step_days = self.val_window_days  # Fixed window
        
        for i in range(self.n_splits):
            if self.strategy == SplitStrategy.ANCHORED:
                # Growing training window
                train_start = start_date
            else:
                # Fixed or sliding training window
                train_start = end_date - timedelta(
                    days=(self.n_splits - i) * step_days + self.train_window_days
                )
            
            train_end = train_start + timedelta(days=self.train_window_days)
            val_start = train_end + timedelta(days=self.gap_days)
            val_end = val_start + timedelta(days=self.val_window_days)
            
            if val_end > end_date:
                break
            
            test_start = None
            test_end = None
            if self.test_window_days > 0:
                test_start = val_end + timedelta(days=self.gap_days)
                test_end = test_start + timedelta(days=self.test_window_days)
                if test_end > end_date:
                    test_start = None
                    test_end = None
            
            fold = CVFold(
                fold_id=len(folds),
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
            
            folds.append(fold)
        
        return folds
    
    def _validate_data(self, data: pd.DataFrame, date_column: str) -> bool:
        """Validate input data format"""
        if data.empty:
            logger.error("Empty DataFrame provided")
            return False
        
        if date_column not in data.columns:
            logger.error(f"Date column '{date_column}' not found in DataFrame")
            return False
        
        return True
    
    def get_fold_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all folds"""
        if not self.folds:
            return pd.DataFrame()
        
        stats = []
        for fold in self.folds:
            stats.append({
                'fold': fold.fold_id,
                'train_start': fold.train_start.date(),
                'train_end': fold.train_end.date(),
                'train_size': fold.train_size,
                'val_start': fold.val_start.date(),
                'val_end': fold.val_end.date(),
                'val_size': fold.val_size,
                'test_size': fold.test_size,
            })
        
        return pd.DataFrame(stats)


class TimeSeriesStratifiedSplit:
    """
    Alternative: Time-series stratified split
    
    Ensures similar distribution of features across folds while maintaining
    temporal integrity.
    """
    
    def __init__(self, n_splits: int = 5, random_state: Optional[int] = None):
        """
        Initialize stratified time-series splitter
        
        Args:
            n_splits: Number of folds
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """
        Generate stratified time-series splits
        
        Args:
            X: Feature matrix with time index
            y: Optional target variable
            groups: Optional group labels (e.g., tickers)
            
        Yields:
            (train_indices, val_indices) tuples
        """
        if X.index.name != 'date' and not hasattr(X.index, 'year'):
            raise ValueError("X must have a datetime index or 'date' index")
        
        # Group by year/month for stratification
        time_groups = X.index.to_period('M')
        unique_periods = time_groups.unique()
        
        # Split periods into folds
        n_periods = len(unique_periods)
        fold_size = n_periods // self.n_splits
        
        for fold in range(self.n_splits):
            fold_start = fold * fold_size
            fold_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_periods
            
            val_periods = set(unique_periods[fold_start:fold_end])
            
            train_indices = np.where(~time_groups.isin(val_periods))[0]
            val_indices = np.where(time_groups.isin(val_periods))[0]
            
            yield train_indices, val_indices


class CrossValidationAnalyzer:
    """Analyze cross-validation splits for quality and characteristics"""
    
    @staticmethod
    def analyze_fold_overlap(folds: List[CVFold]) -> Dict[str, Any]:
        """Check for temporal overlap between folds"""
        overlaps = []
        
        for i, fold1 in enumerate(folds):
            for fold2 in folds[i+1:]:
                # Check train-val overlap
                if fold1.train_end >= fold2.train_start:
                    overlaps.append({
                        'fold1': fold1.fold_id,
                        'fold2': fold2.fold_id,
                        'type': 'train-train',
                        'overlap_days': (fold1.train_end - fold2.train_start).days
                    })
        
        return {
            'has_overlap': len(overlaps) > 0,
            'overlaps': overlaps,
            'is_valid': len(overlaps) == 0
        }
    
    @staticmethod
    def analyze_data_leakage(
        df: pd.DataFrame,
        folds: List[CVFold],
        date_column: str = 'date'
    ) -> Dict[str, Any]:
        """
        Detect potential data leakage between train/val splits
        
        Args:
            df: Data frame with temporal data
            folds: List of CV folds
            date_column: Name of date column
            
        Returns:
            Analysis results
        """
        leakage_issues = []
        df[date_column] = pd.to_datetime(df[date_column])
        
        for fold in folds:
            # Check that validation starts after training ends
            if fold.val_start <= fold.train_end:
                leakage_issues.append({
                    'fold': fold.fold_id,
                    'issue': 'Validation starts before training ends',
                    'severity': 'CRITICAL'
                })
            
            # Check for gap
            gap = (fold.val_start - fold.train_end).days
            if gap < 1:
                leakage_issues.append({
                    'fold': fold.fold_id,
                    'issue': f'No gap between train/val (gap={gap})',
                    'severity': 'HIGH'
                })
        
        return {
            'has_leakage': len(leakage_issues) > 0,
            'issues': leakage_issues,
            'total_folds': len(folds),
            'valid_folds': sum(1 for i in leakage_issues if i['severity'] != 'CRITICAL')
        }
    
    @staticmethod
    def analyze_sample_distribution(
        folds: List[CVFold],
        samples_per_day: int = 1
    ) -> Dict[str, Any]:
        """Analyze sample distribution across folds"""
        total_train = sum(fold.train_size for fold in folds)
        total_val = sum(fold.val_size for fold in folds)
        
        avg_train = total_train / len(folds) if folds else 0
        avg_val = total_val / len(folds) if folds else 0
        
        return {
            'total_folds': len(folds),
            'total_train_samples': total_train,
            'total_val_samples': total_val,
            'avg_train_per_fold': avg_train,
            'avg_val_per_fold': avg_val,
            'train_val_ratio': total_train / total_val if total_val > 0 else 0,
            'folds': [
                {
                    'fold': f.fold_id,
                    'train': f.train_size,
                    'val': f.val_size,
                    'ratio': f.train_size / f.val_size if f.val_size > 0 else 0
                }
                for f in folds
            ]
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Create temporal cross-validation splits
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'ticker': 'AAPL',
        'close': np.random.uniform(100, 200, len(dates)),
        'volume': np.random.randint(1000000, 10000000, len(dates)),
    })
    
    # Initialize validator
    cv = TemporalCrossValidator(
        n_splits=5,
        train_window_days=252,
        val_window_days=63,
        strategy=SplitStrategy.TEMPORAL
    )
    
    # Generate splits
    splits = cv.split(data, date_column='date')
    
    print(f"\nGenerated {len(splits)} CV folds:")
    print(cv.get_fold_statistics())
    
    # Analyze folds
    analyzer = CrossValidationAnalyzer()
    print("\nFold Overlap Analysis:")
    print(analyzer.analyze_fold_overlap(cv.folds))
    
    print("\nData Leakage Analysis:")
    print(analyzer.analyze_data_leakage(data, cv.folds))
    
    print("\nSample Distribution:")
    print(analyzer.analyze_sample_distribution(cv.folds))
