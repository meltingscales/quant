"""
MutationEngine — evolutionary search for high-performing (ticker, strategy, params) combinations.

WARNING: Results are purely in-sample. A high Sharpe ratio found by this engine is
almost certainly inflated by multiple comparisons — you've tried hundreds of combinations
and kept the best-looking one. Always validate survivors on out-of-sample data.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from ..engine import Backtester
from .. import metrics as m


# ---------------------------------------------------------------------------
# Strategy specification
# ---------------------------------------------------------------------------

@dataclass
class ParamRange:
    """Defines the valid range and type for a single strategy parameter."""
    lo: Any
    hi: Any

    @property
    def is_int(self) -> bool:
        return isinstance(self.lo, int)

    def sample(self, rng: random.Random) -> Any:
        if self.is_int:
            return rng.randint(self.lo, self.hi)
        return round(rng.uniform(self.lo, self.hi), 2)

    def mutate(self, value: Any, rng: random.Random, strength: float = 0.15) -> Any:
        span = self.hi - self.lo
        if self.is_int:
            delta = max(1, int(span * strength))
            new_v = value + rng.randint(-delta, delta)
            return max(self.lo, min(self.hi, int(new_v)))
        delta = span * strength
        new_v = value + rng.uniform(-delta, delta)
        return round(max(self.lo, min(self.hi, float(new_v))), 2)


@dataclass
class StrategySpec:
    """
    Describes a strategy class and the valid range of its parameters.

    Args:
        name:        human-readable name for display
        cls:         the Strategy subclass
        ranges:      dict mapping param name → ParamRange
        constraints: optional list of callables(params) → bool.
                     If any returns False the params are invalid and resampled.
    """
    name: str
    cls: type
    ranges: dict[str, ParamRange]
    constraints: list = field(default_factory=list)

    def sample_params(self, rng: random.Random, max_attempts: int = 20) -> dict:
        for _ in range(max_attempts):
            params = {k: v.sample(rng) for k, v in self.ranges.items()}
            if all(c(params) for c in self.constraints):
                return params
        # Fallback: return unconstrained sample (strategy constructor will raise)
        return {k: v.sample(rng) for k, v in self.ranges.items()}

    def mutate_params(self, params: dict, rng: random.Random,
                      strength: float = 0.15, max_attempts: int = 20) -> dict:
        for _ in range(max_attempts):
            mutated = {k: self.ranges[k].mutate(v, rng, strength) for k, v in params.items()}
            if all(c(mutated) for c in self.constraints):
                return mutated
        return params  # return original if we can't satisfy constraints

    def make_strategy(self, params: dict):
        return self.cls(**params)


# ---------------------------------------------------------------------------
# Default seed specs (used when MutationEngine is created without custom specs)
# ---------------------------------------------------------------------------

def default_strategy_specs() -> list[StrategySpec]:
    from ..strategies import SMACrossover, BollingerBands, DonchianBreakout
    from ..strategies import TimeSeriesMomentum, RSITrendFilter

    return [
        StrategySpec(
            name='SMACrossover',
            cls=SMACrossover,
            ranges={
                'fast': ParamRange(5, 60),
                'slow': ParamRange(20, 250),
            },
            constraints=[lambda p: p['fast'] < p['slow'] - 5],
        ),
        StrategySpec(
            name='BollingerBands',
            cls=BollingerBands,
            ranges={
                'window':  ParamRange(10, 60),
                'num_std': ParamRange(1.0, 3.5),
            },
        ),
        StrategySpec(
            name='DonchianBreakout',
            cls=DonchianBreakout,
            ranges={
                'entry': ParamRange(10, 80),
                'exit_': ParamRange(5, 40),
            },
            constraints=[lambda p: p['exit_'] < p['entry'] - 3],
        ),
        StrategySpec(
            name='TimeSeriesMomentum',
            cls=TimeSeriesMomentum,
            ranges={
                'lookback': ParamRange(63, 504),
                'skip':     ParamRange(0, 42),
            },
        ),
        StrategySpec(
            name='RSITrendFilter',
            cls=RSITrendFilter,
            ranges={
                'rsi_window': ParamRange(7, 28),
                'oversold':   ParamRange(20.0, 40.0),
                'trend_sma':  ParamRange(50, 300),
            },
        ),
    ]


DEFAULT_TICKERS = [
    'SPY', 'QQQ', 'IWM',          # broad market
    'AAPL', 'MSFT', 'NVDA',       # large cap tech
    'GOOGL', 'AMZN', 'META',      # large cap tech cont.
    'JPM', 'XOM', 'JNJ',          # financials, energy, healthcare
]


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    ticker: str
    spec: StrategySpec
    params: dict

    @property
    def label(self) -> str:
        param_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f'{self.ticker} | {self.spec.name}({param_str})'


# ---------------------------------------------------------------------------
# MutationEngine
# ---------------------------------------------------------------------------

class MutationEngine:
    """
    Evolutionary search over (ticker, strategy, params) combinations.

    Each generation:
      1. Evaluate the current population via backtest
      2. Keep the top `top_k` individuals by fitness
      3. Generate the next population by mutating survivors + adding fresh random individuals

    Args:
        tickers:          list of ticker symbols to search over
        strategy_specs:   list of StrategySpec describing the search space
        start, end:       date range for backtests (YYYY-MM-DD strings)
        initial_capital:  starting portfolio value
        commission:       per-trade commission as fraction of trade value
        min_trades:       discard individuals with fewer trades than this
        fitness:          scoring metric — 'sharpe', 'calmar', or 'total_return'
        seed:             random seed for reproducibility
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        strategy_specs: list[StrategySpec] | None = None,
        start: str = '2015-01-01',
        end: str = '2025-01-01',
        initial_capital: float = 10_000,
        commission: float = 0.001,
        min_trades: int = 5,
        fitness: str = 'sharpe',
        seed: int = 42,
    ):
        self.tickers = tickers or DEFAULT_TICKERS
        self.specs = strategy_specs or default_strategy_specs()
        self.start = start
        self.end = end
        self.initial_capital = initial_capital
        self.commission = commission
        self.min_trades = min_trades
        self.fitness_metric = fitness
        self.rng = random.Random(seed)
        self._prices: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_prices(self) -> None:
        """Download price data for all tickers (called once before run())."""
        print(f'Downloading prices for {len(self.tickers)} tickers...')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = yf.download(
                self.tickers, start=self.start, end=self.end,
                auto_adjust=True, progress=False,
            )
        closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
        for ticker in self.tickers:
            if ticker in closes.columns:
                s = closes[ticker].dropna()
                if len(s) > 0:
                    self._prices[ticker] = s
        print(f'  Loaded {len(self._prices)} tickers successfully.')

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _fitness(self, result) -> float:
        if self.fitness_metric == 'sharpe':
            rets = result.equity.pct_change().dropna()
            return float(m.sharpe_ratio(rets))
        if self.fitness_metric == 'calmar':
            cagr = m.cagr(result.equity)
            mdd = abs(m.max_drawdown(result.equity))
            return cagr / mdd if mdd > 0 else 0.0
        if self.fitness_metric == 'total_return':
            return float(result.equity.iloc[-1] / result.equity.iloc[0] - 1)
        raise ValueError(f'Unknown fitness metric: {self.fitness_metric}')

    def _evaluate(self, ind: Individual) -> dict | None:
        prices = self._prices.get(ind.ticker)
        if prices is None:
            return None
        try:
            strategy = ind.spec.make_strategy(ind.params)
            bt = Backtester(prices, strategy, self.initial_capital, self.commission)
            result = bt.run()
        except Exception:
            return None

        if len(result.trades) < self.min_trades:
            return None

        score = self._fitness(result)
        if not np.isfinite(score):
            return None

        rets = result.equity.pct_change().dropna()
        return {
            'label':        ind.label,
            'ticker':       ind.ticker,
            'strategy':     ind.spec.name,
            'params':       ind.params,
            self.fitness_metric: round(score, 4),
            'total_return': f"{result.equity.iloc[-1] / result.equity.iloc[0] - 1:.1%}",
            'sharpe':       round(float(m.sharpe_ratio(rets)), 3),
            'max_drawdown': f"{m.max_drawdown(result.equity):.1%}",
            'cagr':         f"{m.cagr(result.equity):.1%}",
            'n_trades':     len(result.trades),
            '_result':      result,   # kept for plotting; stripped in final df
            '_score':       score,
        }

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def _random_individual(self) -> Individual:
        ticker = self.rng.choice(self.tickers)
        spec   = self.rng.choice(self.specs)
        params = spec.sample_params(self.rng)
        return Individual(ticker=ticker, spec=spec, params=params)

    def _mutate(self, ind: Individual) -> Individual:
        roll = self.rng.random()
        if roll < 0.5:
            # Mutate parameters
            new_params = ind.spec.mutate_params(ind.params, self.rng)
            return Individual(ticker=ind.ticker, spec=ind.spec, params=new_params)
        elif roll < 0.75:
            # Swap ticker, keep strategy + params
            new_ticker = self.rng.choice(self.tickers)
            return Individual(ticker=new_ticker, spec=ind.spec, params=dict(ind.params))
        else:
            # Completely fresh individual
            return self._random_individual()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        generations: int = 3,
        population_size: int = 60,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Run the evolutionary search.

        Args:
            generations:     number of generations to run
            population_size: individuals evaluated per generation
            top_k:           survivors carried into the next generation

        Returns:
            pd.DataFrame of all evaluated individuals, sorted by fitness (best first).
            Includes columns: ticker, strategy, params, sharpe, total_return,
            max_drawdown, cagr, n_trades, and a '_result' column for plotting.
        """
        if not self._prices:
            self.load_prices()

        all_records: list[dict] = []
        population: list[Individual] = [self._random_individual() for _ in range(population_size)]

        for gen in range(1, generations + 1):
            print(f'Generation {gen}/{generations} — evaluating {len(population)} individuals...')
            records = []
            for ind in population:
                rec = self._evaluate(ind)
                if rec is not None:
                    rec['generation'] = gen
                    records.append(rec)

            valid = sorted(records, key=lambda r: r['_score'], reverse=True)
            print(f'  {len(valid)} valid | top {self.fitness_metric}: {valid[0]["_score"]:.3f} '
                  f'({valid[0]["label"]})' if valid else '  0 valid individuals')
            all_records.extend(valid)

            if gen == generations:
                break

            # Survivors → mutated offspring + fresh random fill
            survivors = [
                Individual(ticker=r['ticker'], spec=next(s for s in self.specs if s.name == r['strategy']),
                           params=r['params'])
                for r in valid[:top_k]
            ]
            next_pop: list[Individual] = []
            # Each survivor spawns offspring
            for sur in survivors:
                n_offspring = max(1, (population_size - top_k) // len(survivors))
                for _ in range(n_offspring):
                    next_pop.append(self._mutate(sur))
            # Pad with fresh randoms
            while len(next_pop) < population_size:
                next_pop.append(self._random_individual())
            population = next_pop[:population_size]

        # Build final DataFrame (drop internal fields)
        df = pd.DataFrame(all_records)
        df = df.sort_values('_score', ascending=False).reset_index(drop=True)
        display_cols = ['ticker', 'strategy', 'params', 'sharpe', 'total_return',
                        'max_drawdown', 'cagr', 'n_trades', 'generation', '_result']
        return df[[c for c in display_cols if c in df.columns]]
