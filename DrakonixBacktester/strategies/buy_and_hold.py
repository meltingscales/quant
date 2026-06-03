import pandas as pd
from ..strategy import Strategy


class BuyAndHold(Strategy):
    """
    Baseline strategy: buy on the first bar and hold indefinitely.

    Used as a benchmark to compare active strategies against.
    If your strategy can't beat this, it's not adding value.
    """

    def __init__(self):
        self._bought = False

    def generate_signal(self, prices: pd.Series) -> int:
        if not self._bought:
            self._bought = True
            return 1
        return 0

    def reset(self):
        self._bought = False
