# Learning Curriculum - Intermediate Level

**Skill Level:** Intermediate (Some ML experience, new to production systems)  
**Duration:** 8 Weeks (20-25 hours per week)  
**Objective:** Build a deep understanding of production ML and stock price prediction systems

---

## Overview

This curriculum bridges the gap between tutorial-level code and production systems. You'll learn not just *how* to build models, but *how to build them right*.

**Success Criteria:**
- Understand data leakage and how to prevent it
- Implement proper time series validation
- Build a complete ML pipeline from data to serving
- Write production-quality code with proper testing
- Deploy and monitor an ML system

---

## Week 1: Time Series Foundations & Data Leakage

**Goal:** Understand why tutorial approaches fail and correct principles

### Part 1: Time Series Fundamentals (6 hours)

#### Reading (4 hours)
- **Book:** "Forecasting: Principles and Practice" (2nd ed)
  - Chapter 2: Time series graphics (1h)
  - Chapter 3: The forecaster's toolbox (1.5h)
  - Chapter 5: Stationarity and differencing (1.5h)

- **Online:** 
  - StatQuest video: "Correlation vs Autocorrelation" (15 min)
  - StatQuest video: "The ARIMA model" (25 min)

#### Hands-on Exercise 1A: Detect Autocorrelation (2 hours)

```python
# File: exercises/week1/detect_autocorrelation.py

import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Load stock data
data = pd.read_csv('data/AAPL_daily.csv', index_col='Date')
prices = data['Close']

# Task 1: Calculate autocorrelation
acf_values = ...  # TODO: Use statsmodels

# Task 2: Plot ACF
plt.figure(figsize=(12, 4))
plot_acf(prices, lags=40, ax=plt.subplot(121))
plot_pacf(prices, lags=40, ax=plt.subplot(122))
plt.tight_layout()
plt.savefig('outputs/acf_pacf.png')

# Task 3: Interpret results
print("Autocorrelation at lag 1:", acf_values[1])
print("Autocorrelation at lag 5:", acf_values[5])
print("\nWhat do these tell you about predictability?")

# Task 4: Check multiple stocks
for ticker in ['AAPL', 'MSFT', 'GOOGL']:
    # Load and analyze each
    pass

# Success Criteria:
# - Generate ACF plots for 3 stocks
# - Explain why today's price correlates with yesterday's
# - Discuss implications for prediction models
```

**Expected Output:**
```
✅ Autocorrelation at lag 1: 0.998
   (Almost perfect - daily prices are very similar)
✅ Autocorrelation at lag 5: 0.992
✅ Generated plots saved to outputs/
✅ Written explanation (200+ words) on what this means
```

---

### Part 2: Data Leakage Disasters (4 hours)

#### Reading (1.5 hours)
- Article: "Leakage in Data Mining" - https://github.com/ianozsvald/leakage
- Article: "Common Machine Learning Mistakes" - Fast.ai
- Watch: "Data Leakage Explained" (30 min video)

#### Hands-on Exercise 1B: Hunt Down Leakage (2.5 hours)

```python
# File: exercises/week1/identify_leakage.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# SCENARIO 1: Scaler Leakage (Tutorial bug)
print("=" * 60)
print("SCENARIO 1: Scaler Leakage")
print("=" * 60)

data = pd.read_csv('data/AAPL_daily.csv')
prices = data['Close'].values.reshape(-1, 1)

# ❌ WRONG WAY (what tutorial does)
scaler_wrong = MinMaxScaler()
prices_normalized_wrong = scaler_wrong.fit_transform(prices)

# Split after normalization
train_size = int(0.7 * len(prices))
train_wrong = prices_normalized_wrong[:train_size]
test_wrong = prices_normalized_wrong[train_size:]

# ✅ CORRECT WAY
scaler_correct = MinMaxScaler()
scaler_correct.fit(prices[:train_size])  # Fit ONLY on training data
prices_normalized_correct = scaler_correct.transform(prices)

train_correct = prices_normalized_correct[:train_size]
test_correct = prices_normalized_correct[train_size:]

# Compare the distributions
print(f"\n❌ WRONG - Test data min: {test_wrong.min():.3f}, max: {test_wrong.max():.3f}")
print(f"✅ CORRECT - Test data min: {test_correct.min():.3f}, max: {test_correct.max():.3f}")
print("\nWhy the difference? The correct way leaves some test data unnormalized!")
print("This is realistic - in production, we don't know test data scale beforehand.")

# SCENARIO 2: Temporal Leakage
print("\n" + "=" * 60)
print("SCENARIO 2: Using Future Data (Temporal Leakage)")
print("=" * 60)

# ❌ WRONG: Windows include future data
lookback = 60
X_wrong = []
y_wrong = []

for i in range(lookback, len(prices) - 5):
    window = prices[i-lookback:i+5]  # Includes 5 days into future!
    X_wrong.append(window[:lookback])
    y_wrong.append(prices[i])

# ✅ CORRECT: Windows use only past data
X_correct = []
y_correct = []

for i in range(lookback, len(prices)):
    window = prices[i-lookback:i]      # Only historical data
    X_correct.append(window)
    y_correct.append(prices[i])

print(f"\n❌ WRONG - Dataset size: {len(X_wrong)}")
print(f"✅ CORRECT - Dataset size: {len(X_correct)}")
print("\nThe wrong approach uses future data the model will never have access to!")

# SCENARIO 3: Random Split on Time Series
print("\n" + "=" * 60)
print("SCENARIO 3: Random Split Breaks Temporal Order")
print("=" * 60)

dates = pd.date_range('2020-01-01', periods=len(prices))
df = pd.DataFrame({'price': prices.flatten(), 'date': dates})

# ❌ WRONG: Random split shuffles temporal order
X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong, idx_train_wrong, idx_test_wrong = \
    train_test_split(df[['price']], df[['price']], df.index, test_size=0.2, random_state=42)

# Check temporal order
print(f"\n❌ WRONG - Train indices: {sorted(idx_train_wrong[:5])}")
print(f"❌ WRONG - Test indices: {sorted(idx_test_wrong[:5])}")
print("Notice: Test indices are mixed with train indices!")
print("The model can learn from future data during training.")

# ✅ CORRECT: Time-based split
split_idx = int(0.8 * len(df))
train_correct = df.iloc[:split_idx]
test_correct = df.iloc[split_idx:]

print(f"\n✅ CORRECT - Train dates: {train_correct['date'].min()} to {train_correct['date'].max()}")
print(f"✅ CORRECT - Test dates: {test_correct['date'].min()} to {test_correct['date'].max()}")
print("Clean separation! Test data is strictly after training data.")

# Success Criteria Checklist:
print("\n" + "=" * 60)
print("SELF-ASSESSMENT")
print("=" * 60)
criteria = {
    "Explained scaler leakage (why and impact)": False,
    "Showed temporal leakage example": False,
    "Demonstrated random split problem": False,
    "Can define 'leakage' in own words": False,
    "Could explain to colleague": False,
}

# Instructions:
print("Before moving on, answer these questions:")
for criterion in criteria.keys():
    print(f"  [ ] {criterion}")

print("\nIf you can't check all boxes, re-read the learning material.")
```

**Expected Output:**
```
============================================================
SCENARIO 1: Scaler Leakage
============================================================

❌ WRONG - Test data min: 0.000, max: 1.000
✅ CORRECT - Test data min: 0.234, max: 0.876

Why the difference? Test data min and max differ from train!
...
[Continued analysis of all three scenarios]
```

**Deeper Dive Questions:**

1. **Why does the tutorial's approach work in notebooks but fail in production?**
   - In notebook: You always have all data available
   - In production: You only have past data
   - Answer: Your model learned to handle future patterns

2. **Can you fix the tutorial with just small changes?**
   - No - the entire approach needs restructuring
   - Must change: data loading, splitting, validation

3. **What would happen if you deployed the tutorial's model?**
   - Day 1: Predictions look good
   - Month 1: Accuracy drops 20%
   - Month 3: Accuracy drops 50%+
   - Reason: Model was trained on patterns that don't exist in real world

---

### Week 1 Deliverables

**By end of Week 1, you should have:**

✅ **Code created:**
- `exercises/week1/detect_autocorrelation.py` (working)
- `exercises/week1/identify_leakage.py` (working)
- `exercises/week1/leakage_comparison.py` (showing wrong vs right)

✅ **Outputs generated:**
- Autocorrelation plots for 3 stocks
- Comparison tables of leakage impacts
- Before/after code examples

✅ **Understanding:**
- Can explain 3 types of leakage
- Know why temporal order matters
- Understand difference between offline and online evaluation

✅ **Quiz (answers in `exercises/week1/quiz_answers.md`):**
1. What is data leakage?
2. Why doesn't random splitting work for time series?
3. When should you fit your scaler?
4. What's the core principle behind avoiding leakage?
5. How would you explain this to non-technical stakeholder?

**Verification:** `make test-week1` (runs and checks all exercises)

---

## Week 2: Proper Time Series Validation

**Goal:** Implement correct validation strategies that reflect production reality

### Part 1: Cross-Validation for Time Series (5 hours)

#### Reading (2 hours)
- "Forecasting: Principles and Practice" - Chapter 3: Cross-validation
- Scikit-learn documentation: "Time series split"
- Blog: "Why you should avoid K-Fold in time series"

#### Hands-on Exercise 2A: Implement Rolling Window CV (3 hours)

```python
# File: exercises/week2/rolling_window_cv.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

class RollingWindowCV:
    """
    Rolling window cross-validation for time series.
    
    Simulates production: train on past, evaluate on future.
    """
    
    def __init__(self, train_size: int, test_size: int, step_size: int = 1):
        """
        Args:
            train_size: Historical data for training
            test_size: Forward-looking evaluation window
            step_size: How many days to shift each iteration
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(self, X: np.ndarray, y: np.ndarray = None):
        """
        Generate train/test splits for rolling window evaluation.
        
        Yields:
            (train_idx, test_idx) tuples
        """
        n = len(X)
        
        for i in range(0, n - self.train_size - self.test_size, self.step_size):
            train_idx = np.arange(i, i + self.train_size)
            test_idx = np.arange(i + self.train_size, 
                                i + self.train_size + self.test_size)
            
            yield train_idx, test_idx
    
    def evaluate(self, X, y, model, metric_func):
        """
        Evaluate model using rolling window CV.
        
        Returns:
            List of metrics for each window
        """
        metrics = []
        
        for train_idx, test_idx in self.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train on past data
            model.fit(X_train, y_train)
            
            # Evaluate on future data
            y_pred = model.predict(X_test)
            metric = metric_func(y_test, y_pred)
            metrics.append(metric)
        
        return metrics


# APPLICATION: Compare strategies
print("=" * 70)
print("COMPARING VALIDATION STRATEGIES")
print("=" * 70)

# Load data
data = pd.read_csv('data/AAPL_daily.csv')
prices = data['Close'].values

# Create simple features: lagged prices
def create_features(prices, lookback=10):
    X, y = [], []
    for i in range(lookback, len(prices)):
        X.append(prices[i-lookback:i])
        y.append(prices[i])
    return np.array(X), np.array(y)

X, y = create_features(prices, lookback=10)

# Strategy 1: Random Split (❌ WRONG)
print("\n❌ Strategy 1: Random Split (WRONG)")
print("-" * 70)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse_random = mean_squared_error(y_test, y_pred)

print(f"MSE: {mse_random:.4f}")
print(f"RMSE: {np.sqrt(mse_random):.4f}")
print("⚠️  This metric is inflated! Train data mixed with test temporally.")

# Strategy 2: Single Train/Test Split (⚠️  BETTER but still simplistic)
print("\n⚠️  Strategy 2: Single expanding window split")
print("-" * 70)

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse_split = mean_squared_error(y_test, y_pred)

print(f"MSE: {mse_split:.4f}")
print(f"RMSE: {np.sqrt(mse_split):.4f}")
print("✓ Respects temporal order")
print("✗ Only one test window - may overfit to this period")

# Strategy 3: Rolling Window CV (✅ PRODUCTION-GRADE)
print("\n✅ Strategy 3: Rolling Window Cross-Validation (BEST)")
print("-" * 70)

cv = RollingWindowCV(train_size=200, test_size=50, step_size=10)
model = RandomForestRegressor()

metrics = []
for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metrics.append(np.sqrt(mse))
    
    if i < 3:  # Show first few windows
        print(f"Window {i+1}: Train on {len(train_idx)} samples, "
              f"Test on {len(test_idx)} samples, RMSE: {np.sqrt(mse):.4f}")

print(f"\nAverage RMSE: {np.mean(metrics):.4f}")
print(f"Std Dev: {np.std(metrics):.4f}")
print(f"Min: {np.min(metrics):.4f}, Max: {np.max(metrics):.4f}")
print("✓ Respects temporal order")
print("✓ Tests across multiple time periods")
print("✓ Simulates production: always predict future from past")

# COMPARISON TABLE
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)

comparison_data = {
    'Strategy': ['Random Split', 'Expanding Window', 'Rolling Window'],
    'Respects Order': ['❌ No', '✅ Yes', '✅ Yes'],
    'Multiple Windows': ['❌ No', '❌ No', '✅ Yes'],
    'Production Realistic': ['❌ No', '⚠️  Partial', '✅ Yes'],
    'Metric': [f'{np.sqrt(mse_random):.4f}', f'{np.sqrt(mse_split):.4f}', 
               f'{np.mean(metrics):.4f} ± {np.std(metrics):.4f}'],
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Success Criteria:
print("\n" + "=" * 70)
print("SUCCESS CRITERIA - Check these boxes:")
print("=" * 70)
success_items = [
    ("RollingWindowCV class implemented and working", True),
    ("Can explain why random split fails", False),  # You should answer this
    ("Can explain why rolling window is best", False),
    ("Graph showing metric variation across windows", False),
]

for item, done in success_items:
    status = "✅" if done else "❌"
    print(f"{status} {item}")
```

**Expected Output:**
```
======================================================================
COMPARING VALIDATION STRATEGIES
======================================================================

❌ Strategy 1: Random Split (WRONG)
----------------------------------------------------------------------
MSE: 2.3456
RMSE: 1.5316
⚠️  This metric is inflated! Data leakage in temporal order.

⚠️  Strategy 2: Single expanding window split
----------------------------------------------------------------------
MSE: 4.5678
RMSE: 2.1373
✓ Respects temporal order
✗ Only one test metric - high variance

✅ Strategy 3: Rolling Window Cross-Validation (BEST)
----------------------------------------------------------------------
Window 1: Train on 200 samples, Test on 50 samples, RMSE: 1.8234
Window 2: Train on 200 samples, Test on 50 samples, RMSE: 2.1456
Window 3: Train on 200 samples, Test on 50 samples, RMSE: 1.9345
...
Average RMSE: 2.0234
Std Dev: 0.2145
```

---

### Part 2: Backtesting Framework (5 hours)

#### Hands-on Exercise 2B: Build Backtesting System (5 hours)

```python
# File: exercises/week2/backtest_framework.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class TradeResult:
    """Record of a single trade"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    profit_loss: float
    profit_loss_pct: float
    holding_days: int

class StockBacktester:
    """
    Backtest a prediction model on historical data.
    
    Simulates real trading: buy/sell based on predictions
    """
    
    def __init__(self, prices_df: pd.DataFrame, initial_capital: float = 10000):
        """
        Args:
            prices_df: DataFrame with 'Date' and 'Close' columns
            initial_capital: Starting capital for backtesting
        """
        self.prices_df = prices_df.copy()
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = None  # None, 'LONG', or 'SHORT'
        self.trades: List[TradeResult] = []
        self.portfolio_values = []
    
    def run(self, predictions: np.ndarray, threshold: float = 0.5):
        """
        Run backtest based on predictions.
        
        Args:
            predictions: Array of predicted price changes (0-1)
            threshold: Confidence threshold for trading (0.5 = 50% confidence)
        """
        self.portfolio_values = []
        
        for i, (idx, row) in enumerate(self.prices_df.iterrows()):
            current_price = row['Close']
            
            if i >= len(predictions):
                break
            
            confidence = predictions[i]
            
            # Decision logic
            if confidence > threshold and self.position is None:
                # Buy signal
                shares = self.cash / current_price
                self.position = {
                    'type': 'LONG',
                    'entry_date': row['Date'],
                    'entry_price': current_price,
                    'shares': shares,
                }
                self.cash = 0
            
            elif confidence < (1 - threshold) and self.position is not None:
                # Sell signal
                proceeds = self.position['shares'] * current_price
                profit = proceeds - (self.position['shares'] * self.position['entry_price'])
                
                trade = TradeResult(
                    entry_date=self.position['entry_date'],
                    exit_date=row['Date'],
                    entry_price=self.position['entry_price'],
                    exit_price=current_price,
                    profit_loss=profit,
                    profit_loss_pct=(profit / (self.position['shares'] * self.position['entry_price'])) * 100,
                    holding_days=(pd.Timestamp(row['Date']) - pd.Timestamp(self.position['entry_date'])).days
                )
                
                self.trades.append(trade)
                self.cash = proceeds
                self.position = None
            
            # Calculate portfolio value
            if self.position is not None:
                portfolio_value = self.cash + (self.position['shares'] * current_price)
            else:
                portfolio_value = self.cash
            
            self.portfolio_values.append(portfolio_value)
        
        return self.get_statistics()
    
    def get_statistics(self) -> dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'final_value': self.cash,
                'ROI': 0,
            }
        
        trades_df = pd.DataFrame([
            {
                'profit_loss': t.profit_loss,
                'profit_loss_pct': t.profit_loss_pct,
            }
            for t in self.trades
        ])
        
        winning_trades = (trades_df['profit_loss'] > 0).sum()
        total_trades = len(self.trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_profit': trades_df['profit_loss'].sum(),
            'avg_profit_per_trade': trades_df['profit_loss'].mean(),
            'final_value': self.portfolio_values[-1] if self.portfolio_values else self.cash,
            'ROI': ((self.portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100) 
                   if self.portfolio_values else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak"""
        if not self.portfolio_values:
            return 0
        
        portfolio_array = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max * 100
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of returns"""
        if len(self.portfolio_values) < 2:
            return 0
        
        returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
        if returns.std() == 0:
            return 0
        
        annual_return = np.mean(returns) * 252
        annual_std = np.std(returns) * np.sqrt(252)
        
        return annual_return / annual_std if annual_std > 0 else 0


# USAGE EXAMPLE
print("=" * 70)
print("BACKTESTING EXAMPLE")
print("=" * 70)

# Load data
data = pd.read_csv('data/AAPL_daily.csv', parse_dates=['Date'])

# Generate dummy predictions (in realworld, from your model)
# 0.7 = high confidence price will go up, 0.3 = confidence will go down
predictions = np.random.uniform(0.3, 0.7, len(data))

# Run backtest
backtest = StockBacktester(data, initial_capital=10000)
stats = backtest.run(predictions, threshold=0.6)

# Display results
print("\nBacktest Results:")
print("-" * 70)
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key:.<40} {value:.2f}")
    else:
        print(f"{key:.<40} {value}")

# Success Criteria:
print("\n" + "=" * 70)
print("SUCCESS CRITERIA")
print("=" * 70)
print("✅ Backtester class working")
print("✅ Can generate trade records")
print("✅ Calculates Sharpe ratio")
print("✅ Can backtest predictions")
print("✅ Can compare different thresholds")
```

---

### Week 2 Deliverables

✅ **Code:**
- `rolling_window_cv.py` (working RollingWindowCV class)
- `backtest_framework.py` (working backtest engine)
- Comparison script showing different validation strategies

✅ **Understanding:**
- Know 3 validation approaches and when to use each
- Can explain why rolling window is production-realistic
- Understand backtesting as validation mechanism

✅ **Quiz:**
1. Why is temporal order important in time series?
2. What's the difference between expanding and rolling windows?
3. How many test windows do you need minimum?
4. Why is backtesting better than just looking at RMSE?
5. What are drawbacks of rolling window approach?

---

## Weeks 3-8: Advanced Topics (Detailed Modules)

Due to length constraints, here's the structure for remaining weeks:

### Week 3: Feature Engineering for Financial Data (20 hours)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Statistical features (volatility, momentum, correlation)
- Creating feature pipeline
- Feature interactions and selection

### Week 4: Advanced LSTM & Attention (20 hours)
- LSTM architecture deep dive
- Attention mechanisms explained
- Implementing LSTM with attention in PyTorch
- Bidirectional and multi-head attention

### Week 5: Hyperparameter Tuning (16 hours)
- Grid search vs random search vs Bayesian
- Optuna framework
- Setting up hyperparameter sweeps
- Distributed tuning

### Week 6: Experiment Tracking & Model Registry (16 hours)
- MLflow architecture
- Logging parameters, metrics, artifacts
- Model versioning and serving
- Comparing experiments

### Week 7: Production Deployment (20 hours)
- FastAPI fundamentals
- Request validation with Pydantic
- Docker containerization
- Kubernetes basics

### Week 8: Monitoring & Continuous Improvement (16 hours)
- Data drift detection
- Model performance monitoring
- Automated retraining triggers
- A/B testing framework

---

## Final Project: Complete ML System

**Capstone:** Build end-to-end system for 5 stocks

**Deliverables:**
1. Data ingestion pipeline (exercise)
2. Feature engineering pipeline
3. Model training with proper validation
4. MLflow experiment tracking
5. FastAPI serving server
6. Monitoring dashboard
7. Backtest with real signals
8. Documentation and deployment guide

**Success Criteria:**
- Code passes all unit/integration tests
- Achieves 80%+ test coverage
- Can serve predictions with <100ms latency
- Includes proper error handling and logging
- Follows production best practices

---

## Assessment Criteria

### By End of Week 1:
- [ ] Understand data leakage (can explain 3 types)
- [ ] Know why temporal order matters
- [ ] Can identify leakage in code

### By End of Week 2:
- [ ] Implement rolling window CV
- [ ] Understand backtesting
- [ ] Run proper validation on real stock data

### By End of Week 4:
- [ ] Build and train LSTM model
- [ ] Implement attention layer
- [ ] Train without data leakage

### By End of Week 8:
- [ ] Complete working ML system
- [ ] Deploy to API
- [ ] Monitor in production
- [ ] One full paper on approach/results

---

## Resources & References

### Books (Recommended Reading Order)
1. **"Forecasting: Principles and Practice"** (Free online: otexts.com/fpp2/)
   - Chapters 2, 3, 5, 7, 8, 9 are most relevant

2. **"Deep Learning for Time Series Forecasting"** by Jason Brownlee
   - Free posts on machinelearningmastery.com

3. **"Time Series Analysis"** by Box & Jenkins (Advanced reference)

4. **"The ML Handbook"** by Andriy Burkov (Practical overview)

### Online Courses (Supplementary)
- Coursera: "Sequence Models" (Deep Learning Specialization)
- Fast.ai: "Practical Deep Learning for Coders"
- Stanford CS224N: NLP with Deep Learning (Transformers/Attention)

### Key Papers
- "Attention is All You Need" (Vaswani et al., 2017)
- "An Empirical Evaluation of Generic Convolutional and Recurrent Networks" (Bai et al., 2018)
- "Leakage in Data Mining" (Knowledge of Data Mining)

### Tools to explore
- PyTorch Lightning (better training loops)
- Optuna (hyperparameter tuning)
- MLflow (experiment tracking)
- FastAPI (model serving)
- Prometheus + Grafana (monitoring)

---

## How to Use This Curriculum

1. **Start here:** Begin Week 1 material
2. **Code first:** Do exercises before reading theory
3. **Run and fail:** Let code break, fix, understand
4. **Teach others:** Explain topics to colleagues - this deepens learning
5. **Ask questions:** File issues if stuck
6. **Apply immediately:** Use this in actual projects

---

## Success Checklist

Print this and check off as you progress:

```
WEEK 1:
[ ] Read FPP chapters 2-5
[ ] Explain 3 types of leakage
[ ] Run leakage detection script
[ ] Answer all week 1 quiz questions
[ ] Can identify leakage in tutorial code

WEEK 2:
[ ] Understand temporal validation
[ ] Build rolling window CV class
[ ] Run backtest on 3 stocks
[ ] Compare validation strategies
[ ] Explain production-realistic backtesting

WEEK 3-8:
[ ] Complete feature engineering pipeline
[ ] Build LSTM + attention model
[ ] Set up MLflow tracking
[ ] Deploy to FastAPI
[ ] Create monitoring dashboard

CAPSTONE:
[ ] Complete 5-stock system
[ ] Test coverage > 80%
[ ] Documentation complete
[ ] Deployment guide written
[ ] Ready for production
```

---

## Contact & Support

- Questions: File an issue on GitHub
- Get stuck: Check `docs/guides/` folder
- Code review: Email team@example.com
- Peer learning: Use discussions forum

---

**Status:** Complete and ready for self-study  
**Estimated Time:** 8 weeks, 20-25 hours/week  
**Difficulty:** Intermediate → Advanced

Let me know when you've completed Week 1! 🚀
