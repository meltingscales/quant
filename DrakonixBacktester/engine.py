import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .strategy import Strategy
from . import metrics as m


class BacktestResult:
    """
    Holds the output of a backtest run.

    Attributes:
        equity:  pd.Series — dollar equity curve (DatetimeIndex)
        trades:  pd.DataFrame — trade log with columns [action, price, shares]
    """

    def __init__(self, equity: pd.Series, trades: pd.DataFrame, initial_capital: float):
        self.equity = equity
        self.trades = trades
        self.initial_capital = initial_capital

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    def summary(self) -> pd.Series:
        return m.summary(self.equity)

    def plot(self, title: str = 'Backtest Results', benchmark: pd.Series = None):
        """
        Plot equity curve, drawdown, and trade markers.

        Args:
            title:     chart title
            benchmark: optional pd.Series of a benchmark equity curve to overlay
        """
        fig = plt.figure(figsize=(13, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        # --- Equity curve ---
        ax1.plot(self.equity, color='steelblue', linewidth=1.5, label='Strategy')
        if benchmark is not None:
            # Scale benchmark to same starting capital
            scaled = benchmark / benchmark.iloc[0] * self.initial_capital
            ax1.plot(scaled, color='gray', linewidth=1, linestyle='--',
                     alpha=0.7, label='Benchmark')

        # Trade markers
        if not self.trades.empty:
            buys = self.trades[self.trades['action'] == 'BUY']
            sells = self.trades[self.trades['action'] == 'SELL']
            # Map trade dates to equity values
            buy_equity = self.equity.reindex(buys.index, method='nearest')
            sell_equity = self.equity.reindex(sells.index, method='nearest')
            ax1.scatter(buy_equity.index, buy_equity.values,
                        marker='^', color='green', s=60, zorder=5, label='Buy')
            ax1.scatter(sell_equity.index, sell_equity.values,
                        marker='v', color='red', s=60, zorder=5, label='Sell')

        ax1.set_title(title)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        plt.setp(ax1.get_xticklabels(), visible=False)

        # --- Drawdown ---
        rolling_max = self.equity.cummax()
        drawdown = (self.equity - rolling_max) / rolling_max
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color='red', alpha=0.4, label='Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax2.set_xlabel('Date')

        plt.tight_layout()
        plt.show()


class Backtester:
    """
    Bar-by-bar backtester for daily OHLCV data (close prices only).

    Execution model:
        Signal generated at close of day T is executed at close of day T+1.
        This simulates placing a market-on-close order after the signal fires,
        and avoids look-ahead bias.

    Args:
        prices:          pd.Series of daily closing prices (DatetimeIndex)
        strategy:        Strategy instance
        initial_capital: starting cash in dollars (default 10,000)
        commission:      fraction of trade value charged per trade (default 0.001 = 0.1%)
    """

    def __init__(
        self,
        prices: pd.Series,
        strategy: Strategy,
        initial_capital: float = 10_000,
        commission: float = 0.001,
    ):
        self.prices = prices.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self) -> BacktestResult:
        self.strategy.reset()

        prices = self.prices
        n = len(prices)

        cash = float(self.initial_capital)
        shares = 0.0
        equity_records = []
        trade_records = []

        for i in range(n):
            date = prices.index[i]
            price = float(prices.iloc[i])
            history = prices.iloc[: i + 1]

            equity = cash + shares * price
            equity_records.append((date, equity))

            signal = self.strategy.generate_signal(history)

            # Execute at next bar's close
            if i + 1 >= n:
                continue

            next_date = prices.index[i + 1]
            next_price = float(prices.iloc[i + 1])

            if signal == 1 and cash > 0:
                shares_bought = (cash / next_price) * (1 - self.commission)
                shares += shares_bought
                trade_records.append((next_date, 'BUY', next_price, shares_bought))
                cash = 0.0

            elif signal == -1 and shares > 0:
                proceeds = shares * next_price * (1 - self.commission)
                trade_records.append((next_date, 'SELL', next_price, shares))
                cash += proceeds
                shares = 0.0

        equity_series = pd.Series(
            [v for _, v in equity_records],
            index=pd.DatetimeIndex([d for d, _ in equity_records]),
            name='equity',
        )

        if trade_records:
            trades_df = pd.DataFrame(
                trade_records, columns=['date', 'action', 'price', 'shares']
            ).set_index('date')
        else:
            trades_df = pd.DataFrame(columns=['action', 'price', 'shares'])

        return BacktestResult(equity_series, trades_df, self.initial_capital)
