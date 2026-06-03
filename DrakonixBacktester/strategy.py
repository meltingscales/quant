from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Base class for all DrakonixBacktester strategies.

    Subclass this and implement generate_signal(). The backtester calls
    generate_signal() on every bar, passing only the price history up to
    and including that bar — no future data ever leaks in.

    Signal semantics:
        1  = go long (buy with all available cash)
       -1  = go flat (sell all shares)
        0  = hold current position unchanged

    Stateful strategies (e.g. tracking whether we're in a trade) should
    store state on self and implement reset() to clear it between runs.
    """

    @abstractmethod
    def generate_signal(self, prices: pd.Series) -> int:
        """
        Generate a trading signal from price history.

        Args:
            prices: pd.Series of daily closing prices, oldest first,
                    most recent last. Index is a DatetimeIndex.

        Returns:
            1, -1, or 0.
        """

    def reset(self):
        """Reset any internal state. Called by the backtester before each run."""
