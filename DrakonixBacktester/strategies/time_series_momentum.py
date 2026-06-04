import pandas as pd
from ..strategy import Strategy


class TimeSeriesMomentum(Strategy):
    """
    Binary time series momentum (Moskowitz, Ooi & Pedersen, 2012).

    Go long if the return over `lookback` days (measured `skip` days ago)
    is positive; go flat otherwise.

    Args:
        lookback:  return measurement window in trading days (default 252)
        skip:      days to skip before the lookback window starts (default 21)
                   Skipping ~1 month avoids short-term reversal contamination.
    """

    def __init__(self, lookback: int = 252, skip: int = 21):
        self.lookback = lookback
        self.skip = skip

    def generate_signal(self, prices: pd.Series) -> int:
        if len(prices) < self.lookback + self.skip + 1:
            return 0

        price_now  = float(prices.iloc[-self.skip - 1])
        price_then = float(prices.iloc[-self.lookback - self.skip - 1])
        ret_12m = (price_now - price_then) / price_then

        return 1 if ret_12m > 0 else -1
