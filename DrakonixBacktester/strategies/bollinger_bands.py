import pandas as pd
from ..strategy import Strategy


class BollingerBands(Strategy):
    """
    Mean reversion strategy using Bollinger Bands.

    Logic:
        BUY  when price touches or falls below the lower band (oversold)
        SELL when price returns to the moving average (mean reversion complete)

    This is the opposite philosophy to SMA crossover: it bets that extreme
    moves are temporary and price will revert to its recent mean.

    Args:
        window:   lookback period for the moving average and std dev (default 20)
        num_std:  number of standard deviations for the bands (default 2.0)

    Pitfall awareness:
        Mean reversion strategies perform poorly in trending markets — a stock
        that keeps falling will keep triggering buy signals ("catching a falling
        knife"). Always combine with a trend filter in production.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
        self._in_trade = False

    def generate_signal(self, prices: pd.Series) -> int:
        if len(prices) < self.window:
            return 0

        window_prices = prices.iloc[-self.window:]
        mean = window_prices.mean()
        std = window_prices.std()
        lower_band = mean - self.num_std * std
        price = float(prices.iloc[-1])

        if not self._in_trade and price <= lower_band:
            self._in_trade = True
            return 1  # enter: price touched lower band

        if self._in_trade and price >= mean:
            self._in_trade = False
            return -1  # exit: price reverted to mean

        return 0

    def reset(self):
        self._in_trade = False
