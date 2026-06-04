import pandas as pd
from ..strategy import Strategy


class DonchianBreakout(Strategy):
    """
    Donchian Channel Breakout (Turtle Trading system).

    Entry: price breaks above the highest close of the last `entry` bars.
    Exit:  price falls below the lowest close of the last `exit_` bars.

    Args:
        entry:  lookback for entry channel (default 20)
        exit_:  lookback for exit channel (default 10); must be < entry
    """

    def __init__(self, entry: int = 20, exit_: int = 10):
        if exit_ >= entry:
            raise ValueError(f'exit_ ({exit_}) must be less than entry ({entry})')
        self.entry = entry
        self.exit_ = exit_
        self._in_trade = False

    def generate_signal(self, prices: pd.Series) -> int:
        if len(prices) < self.entry + 1:
            return 0

        price      = float(prices.iloc[-1])
        entry_high = float(prices.iloc[-self.entry - 1:-1].max())
        exit_low   = float(prices.iloc[-self.exit_ - 1:-1].min())

        if not self._in_trade and price > entry_high:
            self._in_trade = True
            return 1

        if self._in_trade and price < exit_low:
            self._in_trade = False
            return -1

        return 0

    def reset(self):
        self._in_trade = False
