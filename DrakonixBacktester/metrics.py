import numpy as np
import pandas as pd


TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized Sharpe ratio, assuming risk-free rate = 0."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction (e.g. -0.35 = -35%)."""
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate."""
    n = len(equity)
    total = equity.iloc[-1] / equity.iloc[0]
    return float(total ** (periods_per_year / n) - 1)


def summary(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    """Return a Series of key performance metrics for an equity curve."""
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    return pd.Series({
        'Total Return':       f'{total_return:.1%}',
        'CAGR':               f'{cagr(equity, periods_per_year):.1%}',
        'Sharpe Ratio':       f'{sharpe_ratio(returns, periods_per_year):.2f}',
        'Max Drawdown':       f'{max_drawdown(equity):.1%}',
        'Ann. Volatility':    f'{returns.std() * np.sqrt(periods_per_year):.1%}',
        'Win Rate':           f'{(returns > 0).mean():.1%}',
    })
