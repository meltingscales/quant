"""
StrategyStore — SQLite persistence layer for DrakonixBacktester.

All evaluated strategies are stored permanently, enabling:
- Seeding future mutation runs from historical best survivors
- Tracking strategy decay as market regimes change
- Querying and comparing results across many runs

Default DB path: ~/.drakonix/strategies.db
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DB_PATH = Path.home() / '.drakonix' / 'strategies.db'

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    config_json TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS evaluations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(run_id),
    generation    INTEGER NOT NULL,
    ticker        TEXT    NOT NULL,
    strategy      TEXT    NOT NULL,
    params_json   TEXT    NOT NULL,
    sharpe        REAL,
    total_return  REAL,
    max_drawdown  REAL,
    cagr          REAL,
    n_trades      INTEGER,
    fitness       REAL    NOT NULL,
    data_start    TEXT,
    data_end      TEXT,
    evaluated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS decay_checks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id   INTEGER NOT NULL REFERENCES evaluations(id),
    window_start    TEXT    NOT NULL,
    window_end      TEXT    NOT NULL,
    sharpe          REAL,
    total_return    REAL,
    n_trades        INTEGER,
    checked_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_fitness  ON evaluations(fitness DESC);
CREATE INDEX IF NOT EXISTS idx_eval_ticker   ON evaluations(ticker);
CREATE INDEX IF NOT EXISTS idx_eval_strategy ON evaluations(strategy);
CREATE INDEX IF NOT EXISTS idx_eval_run      ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_decay_eval    ON decay_checks(evaluation_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct_to_float(value: Any) -> float | None:
    """Convert '12.3%' or 0.123 to a float fraction."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).rstrip('%')) / 100
    except (ValueError, TypeError):
        return None


class StrategyStore:
    """
    Thread-safe SQLite store for strategy evaluations.

    Args:
        db_path: path to the SQLite file. Created (with parent dirs) if absent.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ── internals ────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA foreign_keys=ON')
        return conn

    def _init_schema(self):
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── run lifecycle ────────────────────────────────────────────────────────

    def begin_run(self, config: dict) -> int:
        """Create a new run record. Returns run_id."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                'INSERT INTO runs (started_at, config_json) VALUES (?, ?)',
                (_now(), json.dumps(config, default=str)),
            )
            return cur.lastrowid

    def end_run(self, run_id: int, status: str = 'done'):
        with self._lock, self._connect() as conn:
            conn.execute(
                'UPDATE runs SET ended_at=?, status=? WHERE run_id=?',
                (_now(), status, run_id),
            )

    def orphan_running_runs(self) -> list[int]:
        """
        Mark any runs still in 'running' status as 'killed' (stale from a crash).
        Returns their run_ids.
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM runs WHERE status='running'"
            ).fetchall()
            ids = [r['run_id'] for r in rows]
            if ids:
                conn.execute(
                    f"UPDATE runs SET status='killed', ended_at=? "
                    f"WHERE run_id IN ({','.join('?'*len(ids))})",
                    [_now()] + ids,
                )
            return ids

    # ── write ────────────────────────────────────────────────────────────────

    def save_generation(
        self,
        run_id: int,
        generation: int,
        df: pd.DataFrame,
        data_start: str,
        data_end: str,
    ) -> int:
        """
        Persist all valid rows from an engine generation DataFrame.

        Expected columns: ticker, strategy, params, sharpe, total_return,
                          max_drawdown, cagr, n_trades, _score.
        Returns number of rows inserted.
        """
        rows = []
        ts = _now()
        for _, row in df.iterrows():
            params = row.get('params')
            if not isinstance(params, dict):
                continue
            fitness = row.get('_score') or row.get('sharpe')
            if fitness is None:
                continue
            rows.append((
                run_id,
                generation,
                row.get('ticker', ''),
                row.get('strategy', ''),
                json.dumps(params),
                row.get('sharpe') if isinstance(row.get('sharpe'), float) else None,
                _pct_to_float(row.get('total_return')),
                _pct_to_float(row.get('max_drawdown')),
                _pct_to_float(row.get('cagr')),
                row.get('n_trades'),
                float(fitness),
                data_start,
                data_end,
                ts,
            ))
        if not rows:
            return 0
        with self._lock, self._connect() as conn:
            conn.executemany(
                '''INSERT INTO evaluations
                   (run_id, generation, ticker, strategy, params_json,
                    sharpe, total_return, max_drawdown, cagr, n_trades,
                    fitness, data_start, data_end, evaluated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                rows,
            )
        return len(rows)

    # ── read ────────────────────────────────────────────────────────────────

    def top_survivors(
        self,
        n: int = 20,
        min_fitness: float = 0.0,
        ticker: str | None = None,
        strategy: str | None = None,
    ) -> pd.DataFrame:
        """
        Return top-N strategies by fitness for seeding future populations.
        Deduplicates by (ticker, strategy, params_json) — keeps best row.
        """
        clauses, params = ['fitness >= ?'], [min_fitness]
        if ticker:
            clauses.append('ticker = ?')
            params.append(ticker)
        if strategy:
            clauses.append('strategy = ?')
            params.append(strategy)

        where = ' AND '.join(clauses)
        sql = f"""
            SELECT ticker, strategy, params_json, MAX(fitness) AS fitness,
                   sharpe, total_return, max_drawdown, cagr, n_trades,
                   data_start, data_end
            FROM evaluations
            WHERE {where}
            GROUP BY ticker, strategy, params_json
            ORDER BY fitness DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params + [n]).fetchall()

        records = []
        for r in rows:
            rec = dict(r)
            try:
                rec['params'] = json.loads(rec.pop('params_json'))
            except (json.JSONDecodeError, KeyError):
                continue
            records.append(rec)

        return pd.DataFrame(records) if records else pd.DataFrame()

    def all_results(self, limit: int = 1000) -> pd.DataFrame:
        """Return the most recent `limit` evaluations for display in the GUI."""
        sql = """
            SELECT e.id, e.run_id, e.generation, e.ticker, e.strategy,
                   e.params_json, e.sharpe, e.total_return, e.max_drawdown,
                   e.cagr, e.n_trades, e.fitness, e.data_start, e.data_end,
                   e.evaluated_at
            FROM evaluations e
            ORDER BY e.fitness DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, [limit]).fetchall()

        records = []
        for r in rows:
            rec = dict(r)
            try:
                rec['params'] = json.loads(rec.pop('params_json'))
            except (json.JSONDecodeError, KeyError):
                rec['params'] = {}
            records.append(rec)

        return pd.DataFrame(records) if records else pd.DataFrame()

    def stats(self) -> dict:
        """Summary statistics for status display."""
        sql_evals = 'SELECT COUNT(*) AS n, MAX(fitness) AS best FROM evaluations'
        sql_runs  = "SELECT COUNT(*) AS n FROM runs WHERE status='done'"
        sql_best  = """
            SELECT ticker, strategy, params_json, fitness
            FROM evaluations ORDER BY fitness DESC LIMIT 1
        """
        sql_last = 'SELECT MAX(started_at) AS ts FROM runs'
        with self._connect() as conn:
            ev  = dict(conn.execute(sql_evals).fetchone())
            rn  = dict(conn.execute(sql_runs).fetchone())
            bst = conn.execute(sql_best).fetchone()
            lst = dict(conn.execute(sql_last).fetchone())

        result = {
            'total_evaluations': ev.get('n', 0),
            'completed_runs':    rn.get('n', 0),
            'best_fitness':      ev.get('best'),
            'last_run_at':       lst.get('ts'),
            'best_ticker':       None,
            'best_strategy':     None,
        }
        if bst:
            result['best_ticker']   = bst['ticker']
            result['best_strategy'] = bst['strategy']
        return result

    # ── decay detection ──────────────────────────────────────────────────────

    def save_decay_check(
        self,
        evaluation_id: int,
        window_start: str,
        window_end: str,
        sharpe: float | None,
        total_return: float | None,
        n_trades: int | None,
    ):
        with self._lock, self._connect() as conn:
            conn.execute(
                '''INSERT INTO decay_checks
                   (evaluation_id, window_start, window_end,
                    sharpe, total_return, n_trades, checked_at)
                   VALUES (?,?,?,?,?,?,?)''',
                (evaluation_id, window_start, window_end,
                 sharpe, total_return, n_trades, _now()),
            )

    def decay_report(self, top_n: int = 20) -> pd.DataFrame:
        """
        Join top evaluations with their most recent decay check.
        Returns rows sorted by sharpe_delta ascending (worst decay first).

        Columns: ticker, strategy, params, original_fitness,
                 original_sharpe, decay_sharpe, sharpe_delta,
                 window_start, window_end
        """
        sql = """
            WITH ranked AS (
                SELECT e.id, e.ticker, e.strategy, e.params_json,
                       e.fitness AS original_fitness, e.sharpe AS original_sharpe,
                       ROW_NUMBER() OVER (
                           PARTITION BY e.ticker, e.strategy, e.params_json
                           ORDER BY e.fitness DESC
                       ) AS rn
                FROM evaluations e
            ),
            top_evals AS (SELECT * FROM ranked WHERE rn = 1 ORDER BY original_fitness DESC LIMIT ?),
            latest_decay AS (
                SELECT d.evaluation_id,
                       d.sharpe    AS decay_sharpe,
                       d.window_start, d.window_end,
                       ROW_NUMBER() OVER (
                           PARTITION BY d.evaluation_id ORDER BY d.checked_at DESC
                       ) AS dr
                FROM decay_checks d
            )
            SELECT t.ticker, t.strategy, t.params_json,
                   t.original_fitness, t.original_sharpe,
                   ld.decay_sharpe,
                   (ld.decay_sharpe - t.original_sharpe) AS sharpe_delta,
                   ld.window_start, ld.window_end
            FROM top_evals t
            LEFT JOIN latest_decay ld
                ON ld.evaluation_id = t.id AND ld.dr = 1
            ORDER BY sharpe_delta ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, [top_n]).fetchall()

        records = []
        for r in rows:
            rec = dict(r)
            try:
                rec['params'] = json.loads(rec.pop('params_json'))
            except (json.JSONDecodeError, KeyError):
                rec['params'] = {}
            records.append(rec)

        return pd.DataFrame(records) if records else pd.DataFrame()
