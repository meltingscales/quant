#!/usr/bin/env python3
"""
DrakonixBacktester — Tkinter MutationEngine GUI

Lets you stochastically "grow" a strategy population, browse results,
compare equity curves side-by-side, and validate survivors out-of-sample.

Run from repo root:
    uv run python DrakonixBacktester/mutationengine/tkgui.py
    just gui
"""

import sys
import math
import threading
import queue
from pathlib import Path

# Make the package importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import asksaveasfilename

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd

from DrakonixBacktester.mutationengine.engine import (
    MutationEngine, Individual, default_strategy_specs, DEFAULT_TICKERS,
)
from DrakonixBacktester.engine import Backtester
from DrakonixBacktester import metrics as m


# ── constants ────────────────────────────────────────────────────────────────

PAD = dict(padx=4, pady=3)
RESULTS_COLS = (
    'rank', 'ticker', 'strategy', 'params',
    'sharpe', 'total_return', 'max_drawdown', 'cagr', 'n_trades', 'gen',
)
COL_WIDTHS = {
    'rank': 38, 'ticker': 58, 'strategy': 128, 'params': 230,
    'sharpe': 62, 'total_return': 80, 'max_drawdown': 85,
    'cagr': 58, 'n_trades': 62, 'gen': 38,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _labeled_entry(parent, label: str, var, width: int = 12):
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, padx=8, pady=2)
    ttk.Label(row, text=f'{label}:', width=11).pack(side=tk.LEFT)
    ttk.Entry(row, textvariable=var, width=width).pack(side=tk.LEFT)
    return row


def _labeled_spinbox(parent, label: str, var, lo, hi, width: int = 7):
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, padx=8, pady=2)
    ttk.Label(row, text=f'{label}:', width=11).pack(side=tk.LEFT)
    ttk.Spinbox(row, textvariable=var, from_=lo, to=hi, width=width).pack(side=tk.LEFT)
    return row


def _embed_figure(parent, fig: Figure) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(fig, master=parent)
    NavigationToolbar2Tk(canvas, parent).pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas


def _plot_equity_dd(fig: Figure, result, title: str = '', color: str = 'steelblue'):
    """Draw equity curve + drawdown into a 2-row Figure. Returns (ax1, ax2)."""
    fig.clear()
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    equity = result.equity
    ax1.plot(equity, color=color, linewidth=1.4)
    if title:
        ax1.set_title(title, fontsize=9)
    ax1.set_ylabel('Value ($)')
    ax1.tick_params(labelbottom=False)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    if not result.trades.empty:
        buys  = result.trades[result.trades['action'] == 'BUY']
        sells = result.trades[result.trades['action'] == 'SELL']
        buy_eq  = equity.reindex(buys.index,  method='nearest')
        sell_eq = equity.reindex(sells.index, method='nearest')
        ax1.scatter(buy_eq.index,  buy_eq.values,  marker='^', color='green', s=36, zorder=5)
        ax1.scatter(sell_eq.index, sell_eq.values, marker='v', color='red',   s=36, zorder=5)

    dd = (equity - equity.cummax()) / equity.cummax()
    ax2.fill_between(dd.index, dd.values, 0, color='red', alpha=0.35)
    ax2.set_ylabel('Drawdown')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))
    return ax1, ax2


# ── main GUI class ────────────────────────────────────────────────────────────

class DrakonixTkGUI:
    """
    Tkinter front-end for MutationEngine.

    Layout
    ------
    Left panel  — configuration + run/grow/stop controls
    Right panel — results treeview (top) + chart notebook (bottom)
      Chart tabs: Equity Curve | Compare | Validate
    Status bar  — status text + indeterminate progress bar
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('DrakonixBacktester — MutationEngine')
        root.minsize(1150, 740)

        # ── runtime state ──
        self._engine: MutationEngine | None = None
        self._results: pd.DataFrame = pd.DataFrame()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._queue: queue.Queue = queue.Queue()
        self._grow_gen: int = 0
        self._all_records: list[dict] = []   # accumulated across grow generations

        self._build_menu()
        self._build_layout()
        self._poll_queue()

    # ── menu ────────────────────────────────────────────────────────────────

    def _build_menu(self):
        bar = tk.Menu(self.root)
        self.root.config(menu=bar)

        file_m = tk.Menu(bar, tearoff=False)
        file_m.add_command(label='Export Results CSV…', command=self._export_csv)
        file_m.add_separator()
        file_m.add_command(label='Quit', command=self.root.quit)
        bar.add_cascade(label='File', menu=file_m)

        eng_m = tk.Menu(bar, tearoff=False)
        eng_m.add_command(label='Clear Results', command=self._clear_results)
        eng_m.add_command(label='Deflated Sharpe…', command=self._show_deflated_sharpe)
        bar.add_cascade(label='Engine', menu=eng_m)

        bar.add_command(label='About', command=lambda: messagebox.showinfo(
            'About',
            'DrakonixBacktester MutationEngine GUI\n\n'
            'Evolutionary strategy search across tickers, strategies, and params.\n\n'
            '⚠ Results are in-sample. Always validate survivors OOS.',
        ))

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_layout(self):
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(pw, width=275)
        left.pack_propagate(False)
        pw.add(left, weight=0)

        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        self._build_config(left)
        self._build_right(right)
        self._build_statusbar()

    # ── left: config ─────────────────────────────────────────────────────────

    def _build_config(self, parent):
        # Scrollable inner frame
        canvas = tk.Canvas(parent, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas)
        cw = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(cw, width=e.width))

        f = inner

        # Tickers
        ttk.Label(f, text='Tickers', font=('', 10, 'bold')).pack(anchor='w', **PAD)
        self._ticker_vars: dict[str, tk.BooleanVar] = {}
        tf = ttk.Frame(f)
        tf.pack(fill=tk.X, padx=6)
        for i, ticker in enumerate(DEFAULT_TICKERS):
            var = tk.BooleanVar(value=True)
            self._ticker_vars[ticker] = var
            ttk.Checkbutton(tf, text=ticker, variable=var).grid(
                row=i // 3, column=i % 3, sticky='w', padx=2, pady=1)

        ttk.Separator(f, orient='horizontal').pack(fill=tk.X, pady=5)

        # Strategies
        ttk.Label(f, text='Strategies', font=('', 10, 'bold')).pack(anchor='w', **PAD)
        self._strat_vars: dict[str, tk.BooleanVar] = {}
        for spec in default_strategy_specs():
            var = tk.BooleanVar(value=True)
            self._strat_vars[spec.name] = var
            ttk.Checkbutton(f, text=spec.name, variable=var).pack(anchor='w', padx=10)

        ttk.Separator(f, orient='horizontal').pack(fill=tk.X, pady=5)

        # Date range
        ttk.Label(f, text='Date Range', font=('', 10, 'bold')).pack(anchor='w', **PAD)
        self._start_var = tk.StringVar(value='2015-01-01')
        self._end_var   = tk.StringVar(value='2025-01-01')
        _labeled_entry(f, 'Start', self._start_var)
        _labeled_entry(f, 'End',   self._end_var)

        ttk.Separator(f, orient='horizontal').pack(fill=tk.X, pady=5)

        # Engine params
        ttk.Label(f, text='Engine Parameters', font=('', 10, 'bold')).pack(anchor='w', **PAD)
        self._gen_var      = tk.IntVar(value=3)
        self._pop_var      = tk.IntVar(value=60)
        self._topk_var     = tk.IntVar(value=10)
        self._mintrade_var = tk.IntVar(value=5)
        self._seed_var     = tk.IntVar(value=42)
        self._fitness_var  = tk.StringVar(value='sharpe')

        _labeled_spinbox(f, 'Generations', self._gen_var,      1,    20)
        _labeled_spinbox(f, 'Pop Size',    self._pop_var,      10,  200)
        _labeled_spinbox(f, 'Top K',       self._topk_var,     3,    30)
        _labeled_spinbox(f, 'Min Trades',  self._mintrade_var, 1,    30)
        _labeled_spinbox(f, 'Seed',        self._seed_var,     0,  9999)

        row = ttk.Frame(f)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text='Fitness:', width=11).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self._fitness_var,
                     values=['sharpe', 'calmar', 'total_return'],
                     width=13, state='readonly').pack(side=tk.LEFT)

        ttk.Separator(f, orient='horizontal').pack(fill=tk.X, pady=5)

        # Run controls
        btn = ttk.Frame(f)
        btn.pack(fill=tk.X, padx=8, pady=4)
        for text, cmd in [('Run', self._on_run), ('Grow', self._on_grow), ('Stop', self._on_stop)]:
            ttk.Button(btn, text=text, command=cmd).pack(
                side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Separator(f, orient='horizontal').pack(fill=tk.X, pady=5)

        # Live stats
        self._stat_gen  = tk.StringVar(value='Generation: —')
        self._stat_eval = tk.StringVar(value='Evaluated: 0')
        self._stat_best = tk.StringVar(value='Best Sharpe: —')
        for var in (self._stat_gen, self._stat_eval, self._stat_best):
            ttk.Label(f, textvariable=var, font=('', 9)).pack(anchor='w', padx=10)

    # ── right: results + charts ──────────────────────────────────────────────

    def _build_right(self, parent):
        vpw = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        vpw.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(vpw)
        vpw.add(table_frame, weight=1)

        chart_frame = ttk.Frame(vpw)
        vpw.add(chart_frame, weight=2)

        self._build_results_table(table_frame)
        self._build_chart_tabs(chart_frame)

    def _build_results_table(self, parent):
        ttk.Label(parent, text='Results  (click a row to view equity curve)',
                  font=('', 9)).pack(anchor='w', padx=4, pady=2)

        tv = ttk.Treeview(parent, columns=RESULTS_COLS, show='headings',
                          height=7, selectmode='browse')
        self._tv = tv
        for col in RESULTS_COLS:
            tv.heading(col, text=col.replace('_', ' ').title(),
                       command=lambda c=col: self._sort_by(c))
            tv.column(col, width=COL_WIDTHS.get(col, 80), anchor='center')

        vsb = ttk.Scrollbar(parent, orient='vertical',   command=tv.yview)
        hsb = ttk.Scrollbar(parent, orient='horizontal', command=tv.xview)
        tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tv.pack(fill=tk.BOTH, expand=True, padx=4)

        tv.bind('<<TreeviewSelect>>', lambda _e: self._on_row_selected())

        ctx = tk.Menu(self.root, tearoff=False)
        ctx.add_command(label='View Equity Curve',  command=self._on_row_selected)
        ctx.add_command(label='Validate OOS…',      command=self._on_validate_oos)
        ctx.add_command(label='Cross-Asset Test',   command=self._on_cross_asset)
        self._ctx = ctx
        tv.bind('<Button-3>', self._on_right_click)

    def _build_chart_tabs(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self._nb = nb

        # Tab 1: Equity Curve
        eq_tab = ttk.Frame(nb)
        nb.add(eq_tab, text='Equity Curve')
        self._eq_fig    = Figure(figsize=(8, 4), tight_layout=True)
        self._eq_canvas = _embed_figure(eq_tab, self._eq_fig)

        # Tab 2: Compare
        cmp_tab = ttk.Frame(nb)
        nb.add(cmp_tab, text='Compare')
        self._build_compare_tab(cmp_tab)

        # Tab 3: Validate
        val_tab = ttk.Frame(nb)
        nb.add(val_tab, text='Validate')
        self._build_validate_tab(val_tab)

    def _build_compare_tab(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=8, pady=4)

        self._cmp_left_var  = tk.StringVar()
        self._cmp_right_var = tk.StringVar()

        for label, var, attr in [
            ('Left:',  self._cmp_left_var,  '_cmp_left_cb'),
            ('Right:', self._cmp_right_var, '_cmp_right_cb'),
        ]:
            ttk.Label(ctrl, text=label).pack(side=tk.LEFT, padx=(8, 2))
            cb = ttk.Combobox(ctrl, textvariable=var, width=38, state='readonly')
            cb.pack(side=tk.LEFT)
            setattr(self, attr, cb)

        ttk.Button(ctrl, text='Compare', command=self._on_compare).pack(side=tk.LEFT, padx=10)

        self._cmp_fig    = Figure(figsize=(10, 4), tight_layout=True)
        self._cmp_canvas = _embed_figure(parent, self._cmp_fig)

    def _build_validate_tab(self, parent):
        cfg = ttk.LabelFrame(parent, text='Out-of-Sample Validation', padding=8)
        cfg.pack(fill=tk.X, padx=8, pady=4)

        r1 = ttk.Frame(cfg)
        r1.pack(fill=tk.X, pady=2)
        self._oos_start_var = tk.StringVar(value='2023-01-01')
        self._oos_end_var   = tk.StringVar(value='2025-01-01')
        for lbl, var in [('OOS Start:', self._oos_start_var), ('OOS End:', self._oos_end_var)]:
            ttk.Label(r1, text=lbl).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Entry(r1, textvariable=var, width=12).pack(side=tk.LEFT, padx=(0, 12))

        r2 = ttk.Frame(cfg)
        r2.pack(fill=tk.X, pady=4)
        for lbl, cmd in [
            ('Validate OOS',      self._on_validate_oos),
            ('Cross-Asset Test',  self._on_cross_asset),
            ('Deflated Sharpe…',  self._show_deflated_sharpe),
        ]:
            ttk.Button(r2, text=lbl, command=cmd).pack(side=tk.LEFT, padx=4)

        self._val_fig    = Figure(figsize=(8, 3), tight_layout=True)
        self._val_canvas = _embed_figure(parent, self._val_fig)

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, relief='sunken')
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_var = tk.StringVar(value='Ready.')
        ttk.Label(bar, textvariable=self._status_var, anchor='w').pack(side=tk.LEFT, padx=6)
        self._progress = ttk.Progressbar(bar, mode='indeterminate', length=160)
        self._progress.pack(side=tk.RIGHT, padx=6, pady=2)

    # ── engine helpers ───────────────────────────────────────────────────────

    def _selected_tickers(self) -> list[str]:
        return [t for t, v in self._ticker_vars.items() if v.get()]

    def _selected_specs(self):
        all_specs = default_strategy_specs()
        return [s for s in all_specs if self._strat_vars.get(s.name, tk.BooleanVar(value=True)).get()]

    def _make_engine(self) -> MutationEngine:
        return MutationEngine(
            tickers=self._selected_tickers() or DEFAULT_TICKERS,
            strategy_specs=self._selected_specs() or default_strategy_specs(),
            start=self._start_var.get(),
            end=self._end_var.get(),
            commission=0.001,
            min_trades=self._mintrade_var.get(),
            fitness=self._fitness_var.get(),
            seed=self._seed_var.get(),
        )

    def _busy(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _start_spinner(self, msg: str):
        self._status_var.set(msg)
        self._progress.start(12)

    # ── run / grow / stop ────────────────────────────────────────────────────

    def _on_run(self):
        if self._busy():
            return
        self._stop_event.clear()
        self._engine = self._make_engine()
        self._all_records = []
        self._grow_gen = 0
        self._thread = threading.Thread(
            target=self._run_thread, args=(self._gen_var.get(),), daemon=True)
        self._thread.start()
        self._start_spinner('Running…')

    def _on_grow(self):
        if self._busy():
            return
        self._stop_event.clear()
        if self._engine is None:
            self._engine = self._make_engine()
        self._thread = threading.Thread(target=self._grow_thread, daemon=True)
        self._thread.start()
        self._start_spinner('Growing…')

    def _on_stop(self):
        self._stop_event.set()
        self._status_var.set('Stopping after current generation…')

    # ── background threads ───────────────────────────────────────────────────

    def _run_thread(self, generations: int):
        try:
            self._queue.put(('status', 'Downloading prices…'))
            df = self._engine.run(
                generations=generations,
                population_size=self._pop_var.get(),
                top_k=self._topk_var.get(),
            )
            self._all_records = df.to_dict('records')
            self._queue.put(('results', df))
            self._queue.put(('status', f'Done — {len(df)} individuals evaluated.'))
        except Exception as exc:
            self._queue.put(('error', str(exc)))
        finally:
            self._queue.put(('done', None))

    def _grow_thread(self):
        """
        Continuously evolve the population one generation at a time,
        seeding each from the top survivors of all accumulated results.
        Stops when _stop_event is set.
        """
        try:
            if not self._engine._prices:
                self._queue.put(('status', 'Downloading prices…'))
                self._engine.load_prices()

            pop_size = self._pop_var.get()
            top_k    = self._topk_var.get()

            # Re-hydrate survivors from accumulated records
            specs_by_name = {s.name: s for s in self._engine.specs}

            def _survivors_to_population(records: list[dict]) -> list[Individual]:
                top = sorted(records, key=lambda r: r.get('_score', 0), reverse=True)[:top_k]
                pop: list[Individual] = []
                for r in top:
                    spec = specs_by_name.get(r.get('strategy', ''))
                    if spec:
                        parent = Individual(ticker=r['ticker'], spec=spec, params=r['params'])
                        n = max(1, (pop_size - top_k) // len(top))
                        for _ in range(n):
                            pop.append(self._engine._mutate(parent))
                while len(pop) < pop_size:
                    pop.append(self._engine._random_individual())
                return pop[:pop_size]

            if self._all_records:
                population = _survivors_to_population(self._all_records)
            else:
                population = [self._engine._random_individual() for _ in range(pop_size)]

            gen = self._grow_gen

            while not self._stop_event.is_set():
                gen += 1
                self._queue.put(('status', f'Growing — generation {gen}…'))

                records: list[dict] = []
                for ind in population:
                    if self._stop_event.is_set():
                        break
                    rec = self._engine._evaluate(ind)
                    if rec is not None:
                        rec['generation'] = gen
                        records.append(rec)

                self._all_records.extend(records)

                display_cols = [
                    'ticker', 'strategy', 'params', 'sharpe', 'total_return',
                    'max_drawdown', 'cagr', 'n_trades', 'generation', '_result', '_score',
                ]
                df = pd.DataFrame(self._all_records)
                df = df.sort_values('_score', ascending=False).reset_index(drop=True)
                df = df[[c for c in display_cols if c in df.columns]]

                self._queue.put(('results', df))
                self._queue.put(('gen',     gen))

                population = _survivors_to_population(self._all_records)

            self._grow_gen = gen
            self._queue.put(('status', f'Grow paused at generation {gen}. Press Grow to continue.'))

        except Exception as exc:
            self._queue.put(('error', str(exc)))
        finally:
            self._queue.put(('done', None))

    def _validate_oos_thread(self, row: pd.Series):
        try:
            import warnings, yfinance as yf
            ticker = row['ticker']
            spec   = next((s for s in self._engine.specs if s.name == row['strategy']), None)
            if spec is None:
                self._queue.put(('status', f"Unknown strategy: {row['strategy']}"))
                return
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                raw = yf.download(ticker, start=self._oos_start_var.get(),
                                  end=self._oos_end_var.get(),
                                  auto_adjust=True, progress=False)
            prices = raw['Close'].squeeze().dropna()
            if len(prices) < 50:
                self._queue.put(('status', 'Not enough OOS data.'))
                return
            result = Backtester(prices, spec.make_strategy(row['params']), 10_000, 0.001).run()
            self._queue.put(('oos_result', (row, result)))
        except Exception as exc:
            self._queue.put(('error', str(exc)))
        finally:
            self._queue.put(('done', None))

    def _cross_asset_thread(self, row: pd.Series):
        try:
            spec = next((s for s in self._engine.specs if s.name == row['strategy']), None)
            if spec is None:
                return
            records = []
            for ticker, prices in self._engine._prices.items():
                try:
                    result = Backtester(prices, spec.make_strategy(row['params']), 10_000, 0.001).run()
                    rets = result.equity.pct_change().dropna()
                    records.append({
                        'ticker':       ticker,
                        'sharpe':       round(float(m.sharpe_ratio(rets)), 3),
                        'total_return': f'{result.equity.iloc[-1]/result.equity.iloc[0]-1:.1%}',
                        'max_drawdown': f'{m.max_drawdown(result.equity):.1%}',
                        'n_trades':     len(result.trades),
                        '_result':      result,
                    })
                except Exception:
                    continue
            records.sort(key=lambda r: r['sharpe'], reverse=True)
            self._queue.put(('cross_result', (row, records)))
        except Exception as exc:
            self._queue.put(('error', str(exc)))
        finally:
            self._queue.put(('done', None))

    # ── queue polling ─────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg, data = self._queue.get_nowait()
                if msg == 'status':
                    self._status_var.set(data)
                elif msg == 'results':
                    self._results = data
                    self._refresh_table(data)
                    self._refresh_compare_dropdowns(data)
                    self._refresh_stats(data)
                elif msg == 'gen':
                    self._stat_gen.set(f'Generation: {data}')
                elif msg == 'done':
                    self._progress.stop()
                elif msg == 'oos_result':
                    self._render_oos(*data)
                elif msg == 'cross_result':
                    self._render_cross(*data)
                elif msg == 'error':
                    messagebox.showerror('Engine Error', data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ── table ────────────────────────────────────────────────────────────────

    def _refresh_table(self, df: pd.DataFrame):
        self._tv.delete(*self._tv.get_children())
        for i, row in df.head(300).iterrows():
            param_str = ', '.join(f'{k}={v}' for k, v in (row.get('params') or {}).items())
            sharpe = row.get('sharpe', '')
            self._tv.insert('', 'end', iid=str(i), values=(
                i + 1,
                row.get('ticker', ''),
                row.get('strategy', ''),
                param_str,
                f'{sharpe:.3f}' if isinstance(sharpe, float) else sharpe,
                row.get('total_return', ''),
                row.get('max_drawdown', ''),
                row.get('cagr', ''),
                row.get('n_trades', ''),
                row.get('generation', ''),
            ))

    def _sort_by(self, col: str):
        if self._results.empty:
            return
        col_map = {'gen': 'generation', 'rank': None, 'params': None}
        actual = col_map.get(col, col)
        if actual and actual in self._results.columns:
            try:
                ascending = col not in ('sharpe', 'total_return', 'cagr', 'n_trades')
                self._results = self._results.sort_values(
                    actual, ascending=ascending,
                    key=lambda s: pd.to_numeric(s.astype(str).str.rstrip('%'), errors='coerce'),
                ).reset_index(drop=True)
                self._refresh_table(self._results)
            except Exception:
                pass

    def _refresh_stats(self, df: pd.DataFrame):
        self._stat_eval.set(f'Evaluated: {len(df)}')
        if not df.empty and 'sharpe' in df.columns:
            best = df.iloc[0].get('sharpe', '—')
            self._stat_best.set(
                f'Best Sharpe: {best:.3f}' if isinstance(best, float) else f'Best Sharpe: {best}')

    def _refresh_compare_dropdowns(self, df: pd.DataFrame):
        labels = [
            f"#{i+1} {row['ticker']} | {row['strategy']} (S={row['sharpe']:.2f})"
            for i, row in df.head(30).iterrows()
            if isinstance(row.get('sharpe'), (int, float))
        ]
        for cb, var in [(self._cmp_left_cb, self._cmp_left_var),
                        (self._cmp_right_cb, self._cmp_right_var)]:
            cb['values'] = labels
        if labels:
            self._cmp_left_var.set(labels[0])
            if len(labels) > 1:
                self._cmp_right_var.set(labels[1])

    # ── chart: equity ────────────────────────────────────────────────────────

    def _selected_row(self) -> pd.Series | None:
        sel = self._tv.selection()
        if not sel or self._results.empty:
            return None
        idx = int(sel[0])
        return self._results.iloc[idx] if idx < len(self._results) else None

    def _on_row_selected(self):
        row = self._selected_row()
        if row is None:
            return
        result = row.get('_result')
        if result is None:
            return
        self._nb.select(0)
        param_str = ', '.join(f'{k}={v}' for k, v in (row.get('params') or {}).items())
        title = (
            f"{row.get('ticker')} | {row.get('strategy')} ({param_str})\n"
            f"Sharpe {row.get('sharpe', '?')}  "
            f"Return {row.get('total_return', '?')}  "
            f"MDD {row.get('max_drawdown', '?')}"
        )
        _plot_equity_dd(self._eq_fig, result, title=title)
        self._eq_canvas.draw()

    # ── chart: compare ───────────────────────────────────────────────────────

    def _idx_from_dropdown_label(self, label: str) -> int | None:
        try:
            return int(label.split()[0].lstrip('#')) - 1
        except (ValueError, IndexError):
            return None

    def _on_compare(self):
        if self._results.empty:
            return
        idx_l = self._idx_from_dropdown_label(self._cmp_left_var.get())
        idx_r = self._idx_from_dropdown_label(self._cmp_right_var.get())
        if idx_l is None or idx_r is None:
            return
        row_l, row_r = self._results.iloc[idx_l], self._results.iloc[idx_r]
        res_l, res_r = row_l.get('_result'), row_r.get('_result')
        if res_l is None or res_r is None:
            return

        fig = self._cmp_fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.3)

        for col, (row, res, color) in enumerate([
            (row_l, res_l, 'steelblue'),
            (row_r, res_r, 'darkorange'),
        ]):
            ax1 = fig.add_subplot(gs[0, col])
            ax2 = fig.add_subplot(gs[1, col], sharex=ax1)
            param_str = ', '.join(f'{k}={v}' for k, v in (row.get('params') or {}).items())
            ax1.set_title(f"{row['ticker']} | {row['strategy']}\n({param_str})", fontsize=9)
            equity = res.equity
            ax1.plot(equity, color=color, linewidth=1.4)
            ax1.set_ylabel('Value ($)')
            ax1.tick_params(labelbottom=False)
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
            dd = (equity - equity.cummax()) / equity.cummax()
            ax2.fill_between(dd.index, dd.values, 0, color='red', alpha=0.35)
            ax2.set_ylabel('Drawdown')
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))

        self._cmp_canvas.draw()
        self._nb.select(1)

    # ── validation ───────────────────────────────────────────────────────────

    def _on_validate_oos(self):
        row = self._selected_row()
        if row is None:
            messagebox.showinfo('Select a row', 'Click a row in the results table first.')
            return
        if not self._engine or not self._engine._prices:
            messagebox.showinfo('No data', 'Run the engine first to load price data.')
            return
        self._start_spinner(f"Validating OOS {self._oos_start_var.get()} → {self._oos_end_var.get()}…")
        threading.Thread(target=self._validate_oos_thread, args=(row,), daemon=True).start()

    def _on_cross_asset(self):
        row = self._selected_row()
        if row is None:
            messagebox.showinfo('Select a row', 'Click a row in the results table first.')
            return
        if not self._engine or not self._engine._prices:
            messagebox.showinfo('No data', 'Run the engine first to load price data.')
            return
        self._start_spinner('Running cross-asset test…')
        threading.Thread(target=self._cross_asset_thread, args=(row,), daemon=True).start()

    def _render_oos(self, row: pd.Series, result):
        self._nb.select(2)
        rets   = result.equity.pct_change().dropna()
        sharpe = m.sharpe_ratio(rets)
        param_str = ', '.join(f'{k}={v}' for k, v in (row.get('params') or {}).items())
        title = (
            f"OOS: {row['ticker']} | {row['strategy']} ({param_str})\n"
            f"OOS Sharpe: {sharpe:.2f}  (in-sample: {row.get('sharpe', '?')})  "
            f"OOS period: {self._oos_start_var.get()} → {self._oos_end_var.get()}"
        )
        _plot_equity_dd(self._val_fig, result, title=title, color='teal')
        self._val_canvas.draw()
        ret = result.equity.iloc[-1] / result.equity.iloc[0] - 1
        self._status_var.set(
            f"OOS Sharpe: {sharpe:.3f}  Return: {ret:.1%}  "
            f"MDD: {m.max_drawdown(result.equity):.1%}")

    def _render_cross(self, row: pd.Series, records: list[dict]):
        self._nb.select(2)
        fig = self._val_fig
        fig.clear()
        ax = fig.add_subplot(111)

        tickers = [r['ticker'] for r in records]
        sharpes = [r['sharpe'] for r in records]
        colors  = ['steelblue' if t == row['ticker'] else '#a8c8e8' for t in tickers]

        bars = ax.bar(tickers, sharpes, color=colors, edgecolor='navy', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Sharpe Ratio')
        param_str = ', '.join(f'{k}={v}' for k, v in (row.get('params') or {}).items())
        ax.set_title(
            f"Cross-Asset: {row['strategy']} ({param_str})\n"
            f"Blue = original ticker ({row['ticker']})", fontsize=9)
        ax.tick_params(axis='x', rotation=30)
        for bar, s in zip(bars, sharpes):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * (1 if s >= 0 else -1),
                    f'{s:.2f}', ha='center', va='bottom', fontsize=8)

        self._val_canvas.draw()
        avg = sum(sharpes) / len(sharpes) if sharpes else 0
        self._status_var.set(
            f'Cross-asset done. Avg Sharpe: {avg:.3f} across {len(records)} tickers.')

    def _show_deflated_sharpe(self):
        if self._results.empty:
            messagebox.showinfo('No results', 'Run the engine first.')
            return
        n    = len(self._results)
        best = self._results.iloc[0].get('sharpe', None)
        if not isinstance(best, float):
            return
        factor   = math.sqrt(math.log(n)) if n > 1 else 1.0
        deflated = best / factor
        messagebox.showinfo('Deflated Sharpe Ratio',
            f'Strategies evaluated:  {n}\n'
            f'Top in-sample Sharpe:  {best:.3f}\n'
            f'Deflation factor:      {factor:.3f}  (√log({n}))\n'
            f'Deflated Sharpe:       {deflated:.3f}\n\n'
            f'Rule of thumb: deflated Sharpe > 0.5 is worth OOS validation.\n'
            f'Below that, results are likely noise from multiple comparisons.')

    # ── context menu + file ops ──────────────────────────────────────────────

    def _on_right_click(self, event):
        row_id = self._tv.identify_row(event.y)
        if row_id:
            self._tv.selection_set(row_id)
            self._ctx.post(event.x_root, event.y_root)

    def _export_csv(self):
        if self._results.empty:
            messagebox.showinfo('No results', 'Run the engine first.')
            return
        path = asksaveasfilename(defaultextension='.csv',
                                 filetypes=[('CSV', '*.csv')],
                                 title='Export results')
        if path:
            drop = {'_result', '_score'}
            cols = [c for c in self._results.columns if c not in drop]
            self._results[cols].to_csv(path, index=False)
            self._status_var.set(f'Exported → {path}')

    def _clear_results(self):
        self._results    = pd.DataFrame()
        self._all_records = []
        self._grow_gen   = 0
        self._engine     = None
        self._tv.delete(*self._tv.get_children())
        self._eq_fig.clear();  self._eq_canvas.draw()
        self._cmp_fig.clear(); self._cmp_canvas.draw()
        self._val_fig.clear(); self._val_canvas.draw()
        self._stat_gen.set('Generation: —')
        self._stat_eval.set('Evaluated: 0')
        self._stat_best.set('Best Sharpe: —')
        self._status_var.set('Results cleared.')


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    DrakonixTkGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
