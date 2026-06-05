#!/usr/bin/env python3
"""
MutationDaemon — autonomous strategy discovery engine.

Runs the MutationEngine continuously, saving every generation to SQLite,
seeding each generation from the DB's historical best survivors, and
periodically re-validating top strategies on rolling windows to detect decay.

Usage:
    just daemon
    uv run python DrakonixBacktester/mutationengine/daemon.py
    uv run python DrakonixBacktester/mutationengine/daemon.py --help
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import yfinance as yf
import warnings

from DrakonixBacktester.mutationengine.engine import (
    MutationEngine, Individual, default_strategy_specs, DEFAULT_TICKERS,
)
from DrakonixBacktester.engine import Backtester
from DrakonixBacktester import metrics as m
from DrakonixBacktester.store import StrategyStore


# ── formatting helpers ────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).strftime('%H:%M:%S')


def _log(msg: str):
    print(f'[{_ts()}] {msg}', flush=True)


def _banner(text: str):
    w = 60
    print('─' * w)
    print(f'  {text}')
    print('─' * w, flush=True)


# ── daemon ────────────────────────────────────────────────────────────────────

class MutationDaemon:
    """
    Autonomous evolutionary strategy search with SQLite persistence.

    Each generation:
      1. Evaluate a population of (ticker, strategy, params) individuals
      2. Save results to the DB
      3. Seed the next generation from DB-wide best survivors (not just this run)
      4. Every `decay_interval` generations, re-validate top strategies on
         rolling time windows and log decay to the DB

    Restart safety:
      On startup any run still marked 'running' in the DB is marked 'killed'.
      A new run_id is created. The daemon seeds from all historical survivors,
      so no work is lost between restarts.

    Args:
        db_path:          path to SQLite file (default ~/.drakonix/strategies.db)
        tickers:          list of ticker symbols to search over
        start, end:       in-sample date window for backtests
        pop_size:         individuals per generation
        top_k:            survivors carried to next generation
        fitness:          scoring metric — 'sharpe', 'calmar', 'total_return'
        min_trades:       discard individuals with fewer trades than this
        seed:             base random seed (incremented each generation)
        decay_interval:   run decay checks every N generations (0 = never)
        decay_windows:    list of (start, end) tuples for OOS decay re-validation
        generation_pause: seconds to sleep between generations (0 = run flat-out)
        status_cb:        optional callable(gen, df, stats) for GUI integration
    """

    def __init__(
        self,
        db_path: str | Path = StrategyStore.__init__.__defaults__[0],  # DEFAULT_DB_PATH
        tickers: list[str] | None = None,
        start: str = '2015-01-01',
        end: str   = '2025-01-01',
        pop_size: int = 60,
        top_k: int = 10,
        fitness: str = 'sharpe',
        min_trades: int = 5,
        seed: int = 0,
        decay_interval: int = 10,
        decay_windows: list[tuple[str, str]] | None = None,
        generation_pause: float = 0.0,
        status_cb: Callable | None = None,
    ):
        from DrakonixBacktester.store.db import DEFAULT_DB_PATH
        self._db_path         = db_path if db_path != MutationDaemon.__init__.__defaults__[0] \
                                else DEFAULT_DB_PATH
        self._tickers         = tickers or DEFAULT_TICKERS
        self._start           = start
        self._end             = end
        self._pop_size        = pop_size
        self._top_k           = top_k
        self._fitness         = fitness
        self._min_trades      = min_trades
        self._seed            = seed
        self._decay_interval  = decay_interval
        self._decay_windows   = decay_windows or []
        self._generation_pause = generation_pause
        self._status_cb       = status_cb

        self._store      = StrategyStore(self._db_path)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── control ──────────────────────────────────────────────────────────────

    def start(self):
        """Spawn the daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name='MutationDaemon')
        self._thread.start()

    def stop(self, timeout: float = 10.0):
        """Signal the daemon to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── main loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        orphaned = self._store.orphan_running_runs()
        if orphaned:
            _log(f'Marked {len(orphaned)} stale run(s) as killed: {orphaned}')

        config = {
            'tickers': self._tickers, 'start': self._start, 'end': self._end,
            'pop_size': self._pop_size, 'top_k': self._top_k,
            'fitness': self._fitness, 'min_trades': self._min_trades,
        }
        run_id = self._store.begin_run(config)
        _banner(f'Run #{run_id} started  |  DB: {self._db_path}')
        _log(f'Tickers: {self._tickers}')
        _log(f'Period:  {self._start} → {self._end}')

        specs = default_strategy_specs()
        engine = MutationEngine(
            tickers=self._tickers, strategy_specs=specs,
            start=self._start, end=self._end,
            min_trades=self._min_trades, fitness=self._fitness,
            seed=self._seed,
        )

        _log('Downloading prices…')
        engine.load_prices()
        _log(f'Loaded {len(engine._prices)} tickers.')

        population = self._seed_population(engine)
        gen = 0
        total_saved = 0

        try:
            while not self._stop_event.is_set():
                # Honour pause
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.5)
                if self._stop_event.is_set():
                    break

                gen += 1
                t0 = time.monotonic()
                _log(f'Generation {gen} — evaluating {len(population)} individuals…')

                records: list[dict] = []
                for ind in population:
                    if self._stop_event.is_set():
                        break
                    rec = engine._evaluate(ind)
                    if rec is not None:
                        rec['generation'] = gen
                        records.append(rec)

                if not records:
                    _log('  No valid individuals this generation — refreshing population.')
                    population = [engine._random_individual() for _ in range(self._pop_size)]
                    continue

                df = pd.DataFrame(records)
                df = df.sort_values('_score', ascending=False).reset_index(drop=True)

                saved = self._store.save_generation(run_id, gen, df, self._start, self._end)
                total_saved += saved

                best = df.iloc[0]
                elapsed = time.monotonic() - t0
                _log(
                    f'  {len(records)} valid  |  '
                    f'best {self._fitness}: {best.get("_score", 0):.3f}  '
                    f'({best.get("ticker")} / {best.get("strategy")})  '
                    f'[{elapsed:.1f}s]  |  DB total: {total_saved}'
                )

                if self._status_cb:
                    try:
                        self._status_cb(gen, df, self._store.stats())
                    except Exception:
                        pass

                # Decay check
                if self._decay_interval and gen % self._decay_interval == 0 and self._decay_windows:
                    self._run_decay_checks(engine)

                # Build next population from all-time DB survivors
                population = self._seed_population(engine)

                if self._generation_pause > 0:
                    time.sleep(self._generation_pause)

        except Exception as exc:
            _log(f'ERROR: {exc}')
            import traceback; traceback.print_exc()
            self._store.end_run(run_id, 'error')
            return

        self._store.end_run(run_id, 'done')
        _log(f'Run #{run_id} finished. Total saved to DB: {total_saved}')

    # ── seeding ───────────────────────────────────────────────────────────────

    def _seed_population(self, engine: MutationEngine) -> list[Individual]:
        """Build a population seeded from DB survivors + fresh random fill."""
        survivors_df = self._store.top_survivors(n=self._top_k)
        specs_by_name = {s.name: s for s in engine.specs}

        survivors: list[Individual] = []
        for _, row in survivors_df.iterrows():
            spec = specs_by_name.get(row.get('strategy', ''))
            if spec and isinstance(row.get('params'), dict):
                survivors.append(Individual(
                    ticker=row['ticker'], spec=spec, params=row['params']))

        population: list[Individual] = []
        if survivors:
            n_offspring = max(1, (self._pop_size - self._top_k) // len(survivors))
            for sur in survivors:
                for _ in range(n_offspring):
                    population.append(engine._mutate(sur))

        while len(population) < self._pop_size:
            population.append(engine._random_individual())

        return population[:self._pop_size]

    # ── decay detection ───────────────────────────────────────────────────────

    def _run_decay_checks(self, engine: MutationEngine):
        _log(f'Running decay checks across {len(self._decay_windows)} window(s)…')
        top_df = self._store.top_survivors(n=20)
        specs_by_name = {s.name: s for s in engine.specs}

        # We need the evaluation IDs from the DB
        all_evals = self._store.all_results(limit=500)
        if all_evals.empty:
            return

        for _, row in top_df.iterrows():
            spec = specs_by_name.get(row.get('strategy', ''))
            if spec is None or not isinstance(row.get('params'), dict):
                continue

            # Find the DB evaluation id for this (ticker, strategy, params)
            params_json = json.dumps(row['params'])
            match = all_evals[
                (all_evals['ticker']   == row['ticker']) &
                (all_evals['strategy'] == row['strategy'])
            ]
            if match.empty:
                continue
            eval_id = int(match.iloc[0]['id'])

            for w_start, w_end in self._decay_windows:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        raw = yf.download(row['ticker'], start=w_start, end=w_end,
                                          auto_adjust=True, progress=False)
                    prices = raw['Close'].squeeze().dropna()
                    if len(prices) < 50:
                        continue
                    strategy = spec.make_strategy(row['params'])
                    result = Backtester(prices, strategy, 10_000, 0.001).run()
                    rets = result.equity.pct_change().dropna()
                    sharpe = float(m.sharpe_ratio(rets))
                    ret    = float(result.equity.iloc[-1] / result.equity.iloc[0] - 1)
                    self._store.save_decay_check(
                        eval_id, w_start, w_end, sharpe, ret, len(result.trades))
                    _log(
                        f'  decay check  {row["ticker"]}/{row["strategy"]}  '
                        f'window {w_start}→{w_end}  sharpe={sharpe:.2f}')
                except Exception as exc:
                    _log(f'  decay check failed: {exc}')


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='DrakonixBacktester autonomous strategy discovery daemon.')
    p.add_argument('--db',         default=str(StrategyStore.__init__.__defaults__[0] if
                                    StrategyStore.__init__.__defaults__ else '~/.drakonix/strategies.db'),
                   help='Path to SQLite database (default: ~/.drakonix/strategies.db)')
    p.add_argument('--start',      default='2015-01-01', help='Backtest start date')
    p.add_argument('--end',        default='2025-01-01', help='Backtest end date')
    p.add_argument('--pop-size',   default=60,  type=int, help='Population size per generation')
    p.add_argument('--top-k',      default=10,  type=int, help='Survivors per generation')
    p.add_argument('--fitness',    default='sharpe',
                   choices=['sharpe', 'calmar', 'total_return'])
    p.add_argument('--min-trades', default=5,   type=int)
    p.add_argument('--seed',       default=0,   type=int)
    p.add_argument('--pause',      default=0.0, type=float,
                   help='Seconds to sleep between generations')
    p.add_argument('--decay-interval', default=10, type=int,
                   help='Run decay checks every N generations (0=off)')
    p.add_argument('--tickers',    nargs='+', default=None,
                   help='Tickers to search (default: all 12 seeds)')
    return p.parse_args()


def main():
    args = _parse_args()

    # Default decay windows: last 1 year and last 6 months relative to end date
    from datetime import date, timedelta
    try:
        end_dt = date.fromisoformat(args.end)
    except ValueError:
        end_dt = date.today()
    decay_windows = [
        ((end_dt - timedelta(days=365)).isoformat(), args.end),
        ((end_dt - timedelta(days=182)).isoformat(), args.end),
    ]

    daemon = MutationDaemon(
        db_path=args.db,
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        pop_size=args.pop_size,
        top_k=args.top_k,
        fitness=args.fitness,
        min_trades=args.min_trades,
        seed=args.seed,
        decay_interval=args.decay_interval,
        decay_windows=decay_windows,
        generation_pause=args.pause,
    )

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(sig, _frame):
        _log(f'Signal {sig} received — stopping after current generation…')
        daemon.stop(timeout=30)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    daemon.start()
    _banner('MutationDaemon running. Press Ctrl+C to stop.')

    # Block main thread
    try:
        while daemon.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
