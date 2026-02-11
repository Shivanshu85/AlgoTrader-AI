"""
Data Leakage Detection and Prevention Framework

Identifies and prevents common sources of data leakage in time-series models:
1. Look-ahead bias (using future data in training)
2. Feature leakage (using target-dependent features)
3. Temporal leakage (improper test set ordering)
4. Multi-collinearity leakage (proxy variables)
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from enum import Enum

logger = logging.getLogger(__name__)


class LeakageType(Enum):
    """Types of data leakage"""
    LOOKAHEAD = "look-ahead bias"
    FEATURE = "feature leakage"
    TEMPORAL = "temporal leakage"
    MULTI_COLLINEARITY = "multi-collinearity"
    TARGET_DEPENDENT = "target-dependent feature"
    FUTURE_DATA = "future data in training"


@dataclass
class LeakageIssue:
    """Represents a detected leakage issue"""
    
    issue_type: LeakageType
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    description: str
    evidence: Dict[str, Any]
    affected_columns: List[str]
    recommendation: str
    
    def __str__(self):
        return (
            f"[{self.severity}] {self.issue_type.value}: {self.description}\n"
            f"Affected: {', '.join(self.affected_columns)}\n"
            f"Action: {self.recommendation}"
        )


class LeakageDetector:
    """Detects various forms of data leakage"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize leakage detector
        
        Args:
            significance_level: Statistical significance threshold
        """
        self.significance_level = significance_level
        self.issues: List[LeakageIssue] = []
    
    def detect_lookahead_bias(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        feature_columns: List[str],
        future_days: int = 1
    ) -> List[LeakageIssue]:
        """
        Detect look-ahead bias
        
        Check if features contain future information about target.
        
        Args:
            df: DataFrame with features and target
            date_column: Name of date column
            target_column: Name of target column
            feature_columns: List of feature column names
            future_days: Number of days ahead to check
            
        Returns:
            List of detected leakage issues
        """
        issues = []
        df = df.sort_values(date_column)
        
        # Shift target to future
        df['target_future'] = df[target_column].shift(-future_days)
        
        for feature in feature_columns:
            # Check correlation with future target
            correlation = df[[feature, 'target_future']].corr().iloc[0, 1]
            
            # If strong correlation exists, there's look-ahead bias
            if abs(correlation) > 0.7 and not pd.isna(correlation):
                issues.append(LeakageIssue(
                    issue_type=LeakageType.LOOKAHEAD,
                    severity='CRITICAL',
                    description=f"Feature '{feature}' has {correlation:.3f} correlation with {future_days}-day future target",
                    evidence={'correlation': correlation, 'future_days': future_days},
                    affected_columns=[feature],
                    recommendation=f"Remove feature '{feature}' or use {future_days}-day lagged version"
                ))
        
        df = df.drop('target_future', axis=1)
        return issues
    
    def detect_target_dependent_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        correlation_threshold: float = 0.8
    ) -> List[LeakageIssue]:
        """
        Detect features that are too dependent on target
        
        Features with very high correlation to target may be target-leaking.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            feature_columns: List of feature column names
            correlation_threshold: Correlation threshold for concern
            
        Returns:
            List of detected leakage issues
        """
        issues = []
        
        for feature in feature_columns:
            # Calculate correlation with target
            try:
                corr, pvalue = pearsonr(
                    df[feature].dropna(),
                    df.loc[df[feature].notna(), target_column]
                )
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {feature}: {e}")
                continue
            
            if abs(corr) > correlation_threshold:
                issues.append(LeakageIssue(
                    issue_type=LeakageType.TARGET_DEPENDENT,
                    severity='HIGH' if abs(corr) > 0.9 else 'MEDIUM',
                    description=f"Feature '{feature}' has suspiciously high correlation ({corr:.3f}) with target",
                    evidence={'correlation': corr, 'p_value': pvalue},
                    affected_columns=[feature],
                    recommendation=f"Investigate '{feature}' derivation; may be derived from target or contain target information"
                ))
        
        return issues
    
    def detect_temporal_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        date_column: str = 'date'
    ) -> List[LeakageIssue]:
        """
        Detect temporal leakage between train and validation sets
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            date_column: Name of date column
            
        Returns:
            List of detected leakage issues
        """
        issues = []
        
        train_dates = pd.to_datetime(train_df[date_column])
        val_dates = pd.to_datetime(val_df[date_column])
        
        train_max = train_dates.max()
        val_min = val_dates.min()
        
        # Check if validation dates are in future
        if val_min <= train_max:
            issues.append(LeakageIssue(
                issue_type=LeakageType.TEMPORAL,
                severity='CRITICAL',
                description=f"Validation set contains data from training period",
                evidence={
                    'train_max_date': str(train_max.date()),
                    'val_min_date': str(val_min.date()),
                    'overlap_days': (train_max - val_min).days
                },
                affected_columns=[date_column],
                recommendation="Ensure validation data is strictly after training data"
            ))
        
        # Check minimum gap
        gap = (val_min - train_max).days
        if gap < 1:
            issues.append(LeakageIssue(
                issue_type=LeakageType.TEMPORAL,
                severity='HIGH',
                description=f"No gap between training and validation ({gap} days)",
                evidence={'gap_days': gap},
                affected_columns=[date_column],
                recommendation="Add at least 1-day gap between train and validation to prevent leakage"
            ))
        
        return issues
    
    def detect_multicollinearity_leakage(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        correlation_threshold: float = 0.95
    ) -> List[LeakageIssue]:
        """
        Detect highly correlated features (potential proxy variables)
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            correlation_threshold: Correlation threshold for concern
            
        Returns:
            List of detected leakage issues
        """
        issues = []
        
        # Calculate correlation matrix
        corr_matrix = df[feature_columns].corr().abs()
        
        # Find highly correlated pairs
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                
                if corr_val > correlation_threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    
                    issues.append(LeakageIssue(
                        issue_type=LeakageType.MULTI_COLLINEARITY,
                        severity='MEDIUM',
                        description=f"Features '{col1}' and '{col2}' are highly correlated ({corr_val:.3f})",
                        evidence={'correlation': corr_val},
                        affected_columns=[col1, col2],
                        recommendation=f"Consider removing one of '{col1}' or '{col2}' to reduce redundancy"
                    ))
        
        return issues
    
    def detect_future_data_usage(
        self,
        df: pd.DataFrame,
        date_column: str,
        feature_columns: List[str],
        target_column: str
    ) -> List[LeakageIssue]:
        """
        Detect if features use future data
        
        Check if any feature values appear after the target date.
        
        Args:
            df: DataFrame with temporal data
            date_column: Name of date column
            feature_columns: List of feature column names
            target_column: Name of target column
            
        Returns:
            List of detected leakage issues
        """
        issues = []
        df = df.sort_values(date_column)
        
        for feature in feature_columns:
            # Check if feature has non-null values in future periods
            for i in range(len(df) - 1):
                current_date = df.iloc[i][date_column]
                feature_value = df.iloc[i][feature]
                
                # Look ahead to see if this feature appears to use future data
                if pd.notna(feature_value):
                    future_window = df.iloc[i:min(i+5, len(df))]
                    
                    # If the feature seems to summarize future data, flag it
                    if hasattr(df[feature].iloc[i:], 'expanding'):
                        pass  # This would require more sophisticated detection
        
        return issues
    
    def comprehensive_check(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        date_column: str = 'date'
    ) -> Dict[str, Any]:
        """
        Run comprehensive leakage detection
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            target_column: Name of target column
            feature_columns: List of feature column names
            date_column: Name of date column
            
        Returns:
            Comprehensive analysis results
        """
        self.issues = []
        
        # Run all checks
        self.issues.extend(self.detect_lookahead_bias(
            train_df, date_column, target_column, feature_columns
        ))
        
        self.issues.extend(self.detect_target_dependent_features(
            train_df, target_column, feature_columns
        ))
        
        self.issues.extend(self.detect_temporal_leakage(
            train_df, val_df, date_column
        ))
        
        self.issues.extend(self.detect_multicollinearity_leakage(
            train_df, feature_columns
        ))
        
        self.issues.extend(self.detect_multicollinearity_leakage(
            val_df, feature_columns
        ))
        
        # Categorize by severity
        critical_issues = [i for i in self.issues if i.severity == 'CRITICAL']
        high_issues = [i for i in self.issues if i.severity == 'HIGH']
        medium_issues = [i for i in self.issues if i.severity == 'MEDIUM']
        
        return {
            'total_issues': len(self.issues),
            'critical': len(critical_issues),
            'high': len(high_issues),
            'medium': len(medium_issues),
            'issues': self.issues,
            'has_critical_leakage': len(critical_issues) > 0,
            'is_safe': len(critical_issues) == 0 and len(high_issues) == 0
        }
    
    def get_report(self) -> str:
        """Generate leakage detection report"""
        if not self.issues:
            return "✅ No leakage issues detected!"
        
        report = f"⚠️  Data Leakage Detection Report ({len(self.issues)} issues)\n"
        report += "=" * 60 + "\n\n"
        
        # Group by severity
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            severity_issues = [i for i in self.issues if i.severity == severity]
            if severity_issues:
                report += f"\n{severity} SEVERITY ({len(severity_issues)} issues):\n"
                report += "-" * 40 + "\n"
                for i, issue in enumerate(severity_issues, 1):
                    report += f"\n{i}. {issue}\n"
        
        return report


class DataLeakagePrevention:
    """Best practices for preventing data leakage"""
    
    @staticmethod
    def create_temporal_features(
        df: pd.DataFrame,
        date_column: str,
        lag_periods: List[int],
        rolling_windows: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged and rolling features safely
        
        Ensures no look-ahead bias.
        
        Args:
            df: DataFrame with time-series data
            date_column: Name of date column
            lag_periods: Lag periods for lagged features
            rolling_windows: Window sizes for rolling features
            
        Returns:
            DataFrame with new temporal features
        """
        df = df.sort_values(date_column)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [c for c in numeric_cols if c != date_column]
        
        for lag in lag_periods:
            for col in numeric_cols:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        for window in rolling_windows:
            for col in numeric_cols:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
        
        # Drop rows with NaN from lagging/rolling
        df = df.dropna()
        
        return df
    
    @staticmethod
    def validate_feature_dates(
        df: pd.DataFrame,
        feature_columns: List[str],
        date_column: str,
        target_date_column: str = 'target_date'
    ) -> bool:
        """
        Ensure feature dates don't exceed target dates
        
        Args:
            df: DataFrame with temporal data
            feature_columns: List of feature column names
            date_column: Name of feature date column
            target_date_column: Name of target date column
            
        Returns:
            True if validation passes
        """
        df[date_column] = pd.to_datetime(df[date_column])
        df[target_date_column] = pd.to_datetime(df[target_date_column])
        
        violations = df[df[date_column] > df[target_date_column]]
        
        if len(violations) > 0:
            logger.error(
                f"Found {len(violations)} date violations where "
                f"feature date > target date"
            )
            return False
        
        return True
    
    @staticmethod
    def split_by_date_only(
        df: pd.DataFrame,
        date_column: str,
        train_start: datetime,
        train_end: datetime,
        val_start: datetime,
        val_end: datetime
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data strictly by dates to prevent any overlap
        
        Args:
            df: DataFrame to split
            date_column: Name of date column
            train_start: Training period start
            train_end: Training period end
            val_start: Validation period start (must be > train_end)
            val_end: Validation period end
            
        Returns:
            (train_df, val_df) tuple
        """
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Enforce no overlap
        assert val_start > train_end, "Validation must start after training ends"
        
        train_df = df[
            (df[date_column] >= train_start) &
            (df[date_column] <= train_end)
        ].copy()
        
        val_df = df[
            (df[date_column] >= val_start) &
            (df[date_column] <= val_end)
        ].copy()
        
        return train_df, val_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'price': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'returns': np.random.uniform(-0.05, 0.05, len(dates)),
    })
    
    # Add target: next day returns
    data['target'] = data['returns'].shift(-1)
    
    # Create suspicious feature (has future information)
    data['suspicious_feature'] = data['target'].shift(-1)  # Uses 2-day-ahead target!
    
    # Split data
    split_date = pd.Timestamp('2023-01-01')
    train_df = data[data['date'] < split_date]
    val_df = data[data['date'] >= split_date]
    
    # Detect leakage
    detector = LeakageDetector()
    results = detector.comprehensive_check(
        train_df, val_df,
        target_column='target',
        feature_columns=['volume', 'returns', 'suspicious_feature'],
        date_column='date'
    )
    
    print(f"\n{detector.get_report()}")
    print(f"\nSafe for modeling: {results['is_safe']}")
