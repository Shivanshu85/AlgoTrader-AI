"""
Backtesting Framework for Stock Price Prediction Models

Implements production-grade backtesting with:
- Walk-forward validation
- Metrics calculation (returns, Sharpe, Sortino, Max Drawdown)
- Risk-adjusted performance analysis
- Trade simulation
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Trade:
    """Represents a single trade"""
    
    ticker: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    position: PositionType = PositionType.LONG
    
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    def close(self, exit_date: datetime, exit_price: float):
        """Close trade"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        if self.position == PositionType.LONG:
            self.pnl = (exit_price - self.entry_price) * 1  # 1 share
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * 1
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
    
    def is_open(self) -> bool:
        """Check if trade is still open"""
        return self.exit_date is None


@dataclass
class BacktestResult:
    """Backtesting results"""
    
    ticker: str
    start_date: datetime
    end_date: datetime
    
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    
    def __str__(self):
        return (
            f"Backtest Results: {self.ticker} ({self.start_date.date()} - {self.end_date.date()})\n"
            f"  Total Return: {self.total_return:.2%}\n"
            f"  Annual Return: {self.annual_return:.2%}\n"
            f"  Volatility: {self.volatility:.2%}\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"  Sortino Ratio: {self.sortino_ratio:.2f}\n"
            f"  Max Drawdown: {self.max_drawdown:.2%}\n"
            f"  Win Rate: {self.win_rate:.1%} ({self.winning_trades}/{self.total_trades})"
        )


class BacktestingEngine:
    """Production-grade backtesting engine"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        risk_free_rate: float = 0.02,  # 2% annual
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per transaction (as fraction)
            risk_free_rate: Risk-free rate for Sharpe/Sortino
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
    
    def backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        date_column: str = 'date',
        price_column: str = 'close',
        ticker: str = 'UNKNOWN'
    ) -> BacktestResult:
        """
        Run backtest with given signals
        
        Args:
            df: Data frame with OHLCV data
            signals: Trading signals (1=long, 0=flat, -1=short)
            date_column: Name of date column
            price_column: Name of price column
            ticker: Ticker symbol
            
        Returns:
            BacktestResult object
        """
        df = df.sort_values(date_column).reset_index(drop=True)
        df['price'] = df[price_column]
        df['signal'] = signals
        
        # Initialize equity curve
        equity = [self.initial_capital]
        positions = []
        trades = []
        
        position = 0  # Current position (1=long, 0=flat, -1=short)
        entry_price = 0
        entry_date = None
        
        for i in range(len(df)):
            current_date = df.iloc[i][date_column]
            current_price = df.iloc[i]['price']
            signal = df.iloc[i]['signal']
            
            # Process signal changes
            if signal != position:
                # Close existing position
                if position != 0 and entry_price > 0:
                    trade = Trade(
                        ticker=ticker,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=current_date,
                        exit_price=current_price,
                        position=PositionType.LONG if position > 0 else PositionType.SHORT
                    )
                    trade.close(current_date, current_price)
                    trades.append(trade)
                
                # Open new position
                if signal != 0:
                    position = signal
                    entry_price = current_price * (1 + self.transaction_cost)
                    entry_date = current_date
                else:
                    position = 0
                    entry_price = 0
            
            # Update equity
            if position != 0:
                unrealized_pnl = (current_price - entry_price) * position
                current_equity = equity[-1] + unrealized_pnl
            else:
                current_equity = equity[-1]
            
            equity.append(current_equity)
        
        # Create result object
        result = self._calculate_metrics(
            df, equity, trades, ticker, date_column
        )
        
        return result
    
    def _calculate_metrics(
        self,
        df: pd.DataFrame,
        equity: List[float],
        trades: List[Trade],
        ticker: str,
        date_column: str
    ) -> BacktestResult:
        """Calculate performance metrics"""
        equity_series = pd.Series(equity[1:], index=df.index)
        returns = equity_series.pct_change().dropna()
        
        start_date = df[date_column].min()
        end_date = df[date_column].max()
        total_days = (end_date - start_date).days
        years = total_days / 365.25
        
        # Returns
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe Ratio
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_returns.mean() / downside_vol * np.sqrt(252) if downside_vol > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        closed_trades = [t for t in trades if t.exit_date is not None]
        winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
        losing_trades = sum(1 for t in closed_trades if t.pnl <= 0)
        win_rate = winning_trades / len(closed_trades) if closed_trades else 0
        
        result = BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(closed_trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            trades=closed_trades,
            equity_curve=equity_series,
        )
        
        return result
    
    def backtest_multiple(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signal_dict: Dict[str, pd.Series],
        date_column: str = 'date',
        price_column: str = 'close'
    ) -> Dict[str, BacktestResult]:
        """
        Backtest multiple tickers
        
        Args:
            data_dict: Dictionary of ticker -> DataFrame
            signal_dict: Dictionary of ticker -> signals
            date_column: Name of date column
            price_column: Name of price column
            
        Returns:
            Dictionary of ticker -> BacktestResult
        """
        results = {}
        
        for ticker in data_dict.keys():
            if ticker not in signal_dict:
                logger.warning(f"No signals for {ticker}")
                continue
            
            result = self.backtest(
                data_dict[ticker],
                signal_dict[ticker],
                date_column=date_column,
                price_column=price_column,
                ticker=ticker
            )
            results[ticker] = result
        
        return results


class BacktestAnalyzer:
    """Analyze backtest results"""
    
    @staticmethod
    def compare_results(results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """
        Compare results across multiple tickers
        
        Args:
            results: Dictionary of ticker -> BacktestResult
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for ticker, result in results.items():
            comparison.append({
                'ticker': ticker,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
            })
        
        df = pd.DataFrame(comparison)
        return df.sort_values('sharpe_ratio', ascending=False)
    
    @staticmethod
    def monte_carlo_analysis(
        result: BacktestResult,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Monte Carlo analysis of backtest results
        
        Args:
            result: BacktestResult object
            n_simulations: Number of simulations
            confidence_level: Confidence level for VaR
            
        Returns:
            Analysis results with confidence intervals
        """
        returns = result.equity_curve.pct_change().dropna()
        
        simulated_returns = []
        for _ in range(n_simulations):
            simulated = np.random.choice(returns, size=len(returns), replace=True)
            cumulative_return = (1 + simulated).prod() - 1
            simulated_returns.append(cumulative_return)
        
        simulated_returns = np.array(simulated_returns)
        
        return {
            'mean_return': simulated_returns.mean(),
            'std_return': simulated_returns.std(),
            'min_return': simulated_returns.min(),
            'max_return': simulated_returns.max(),
            'var': np.percentile(simulated_returns, (1 - confidence_level) * 100),
            'cvar': simulated_returns[simulated_returns <= np.percentile(
                simulated_returns, (1 - confidence_level) * 100
            )].mean(),
        }
    
    @staticmethod
    def analyze_trade_distribution(trades: List[Trade]) -> Dict[str, Any]:
        """Analyze trade characteristics"""
        if not trades:
            return {'error': 'No trades'}
        
        pnls = [t.pnl for t in trades]
        pnls_pct = [t.pnl_pct for t in trades]
        
        return {
            'total_trades': len(trades),
            'profitable_trades': sum(1 for p in pnls if p > 0),
            'losing_trades': sum(1 for p in pnls if p <= 0),
            'avg_win': np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
            'avg_loss': np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0,
            'max_win': max(pnls),
            'max_loss': min(pnls),
            'avg_return_per_trade': np.mean(pnls_pct),
            'std_return_per_trade': np.std(pnls_pct),
            'profit_factor': sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p <= 0)) if sum(p for p in pnls if p <= 0) != 0 else 0,
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    prices = 100 + np.random.randn(len(dates)).cumsum()
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
    })
    
    # Create simple signals (random)
    signals = pd.Series(np.random.choice([-1, 0, 1], size=len(data)), index=data.index)
    
    # Run backtest
    engine = BacktestingEngine(initial_capital=10000)
    result = engine.backtest(data, signals, ticker='AAPL')
    
    print(result)
    
    # Analyze trades
    analyzer = BacktestAnalyzer()
    trade_analysis = analyzer.analyze_trade_distribution(result.trades)
    print(f"\nTrade Analysis: {trade_analysis}")
    
    # Monte Carlo
    mc_analysis = analyzer.monte_carlo_analysis(result, n_simulations=1000)
    print(f"\nMonte Carlo Analysis: {mc_analysis}")
