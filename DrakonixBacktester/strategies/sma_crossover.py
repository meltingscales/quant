import pandas as pd
from ..strategy import Strategy


class SMACrossover(Strategy):
    """
    Dual simple moving average crossover — a classic momentum strategy.

    Logic:
        BUY  when the fast SMA crosses above the slow SMA (uptrend starting)
        SELL when the fast SMA crosses below the slow SMA (downtrend starting)

    This strategy is always either long or flat (never short).

    Args:
        fast: lookback window for the fast SMA (default 20 days)
        slow: lookback window for the slow SMA (default 50 days)

    Pitfall awareness:
        SMA crossovers are trend-following — they work well in trending markets
        and get chopped up (many small losses) in sideways markets. Always check
        your backtest period for survivorship/regime bias.
    """

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError(f'fast ({fast}) must be less than slow ({slow})')
        self.fast = fast
        self.slow = slow

    def generate_signal(self, prices: pd.Series) -> int:
        # Need at least slow + 1 bars to compute a crossover (need previous value too)
        if len(prices) < self.slow + 1:
            return 0

        fast_now  = prices.iloc[-self.fast:].mean()
        fast_prev = prices.iloc[-self.fast - 1:-1].mean()
        slow_now  = prices.iloc[-self.slow:].mean()
        slow_prev = prices.iloc[-self.slow - 1:-1].mean()

        # Golden cross: fast crosses above slow
        if fast_prev <= slow_prev and fast_now > slow_now:
            return 1

        # Death cross: fast crosses below slow
        if fast_prev >= slow_prev and fast_now < slow_now:
            return -1

        return 0
