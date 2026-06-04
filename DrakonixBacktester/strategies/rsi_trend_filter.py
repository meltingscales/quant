import pandas as pd
from ..strategy import Strategy


class RSITrendFilter(Strategy):
    """
    RSI mean reversion, gated by a long-term trend filter.

    Entry:  RSI < oversold AND price > trend_sma-day SMA
    Exit:   RSI >= 50
    Filter: no new entries when price is below the trend SMA

    Args:
        rsi_window:   RSI calculation period (default 14)
        oversold:     RSI threshold to trigger entry (default 30)
        trend_sma:    SMA period for trend filter (default 200)
    """

    def __init__(self, rsi_window: int = 14, oversold: float = 30.0, trend_sma: int = 200):
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.trend_sma = trend_sma
        self._in_trade = False

    def _rsi(self, prices: pd.Series) -> float:
        delta = prices.diff().dropna()
        gain = delta.clip(lower=0).rolling(self.rsi_window).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_window).mean()
        rs = gain / loss.replace(0, float('inf'))
        return float(100 - (100 / (1 + rs.iloc[-1])))

    def generate_signal(self, prices: pd.Series) -> int:
        if len(prices) < self.trend_sma + self.rsi_window + 1:
            return 0

        price      = float(prices.iloc[-1])
        sma        = float(prices.iloc[-self.trend_sma:].mean())
        rsi        = self._rsi(prices)
        in_uptrend = price > sma

        if not self._in_trade and in_uptrend and rsi < self.oversold:
            self._in_trade = True
            return 1

        if self._in_trade and rsi >= 50:
            self._in_trade = False
            return -1

        return 0

    def reset(self):
        self._in_trade = False
