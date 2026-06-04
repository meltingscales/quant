"""
MutationEngine GUI — interactive browser for comparing backtest results.

Uses ipywidgets (bundled with Jupyter). Run inside a Jupyter notebook cell:

    from DrakonixBacktester.mutationengine.gui import ResultsBrowser
    browser = ResultsBrowser(results_df)
    browser.show()
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from IPython.display import display


class ResultsBrowser:
    """
    Interactive side-by-side strategy comparison widget.

    Args:
        results:         DataFrame returned by MutationEngine.run()
        initial_capital: used to normalise equity curves to a common start
        top_n:           how many strategies to offer in the dropdowns (default 20)
    """

    def __init__(self, results: pd.DataFrame, initial_capital: float = 10_000, top_n: int = 20):
        if '_result' not in results.columns:
            raise ValueError("DataFrame must contain '_result' column (BacktestResult objects).")

        self.results = results.dropna(subset=['_result']).head(top_n).reset_index(drop=True)
        self.initial_capital = initial_capital
        self._build()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _labels(self) -> list[str]:
        return [
            f"#{i+1}  {row['ticker']} | {row['strategy']}  "
            f"[Sharpe {row['sharpe']:.2f}, {row['total_return']}, MDD {row['max_drawdown']}]"
            for i, row in self.results.iterrows()
        ]

    def _build(self):
        labels = self._labels()

        self._dd_left = widgets.Dropdown(
            options=labels, value=labels[0],
            description='Left:', layout=widgets.Layout(width='48%'),
            style={'description_width': '40px'},
        )
        self._dd_right = widgets.Dropdown(
            options=labels, value=labels[min(1, len(labels) - 1)],
            description='Right:', layout=widgets.Layout(width='48%'),
            style={'description_width': '44px'},
        )
        self._btn = widgets.Button(
            description='Compare', button_style='primary',
            layout=widgets.Layout(width='120px'),
        )
        self._out = widgets.Output()

        self._btn.on_click(self._on_compare)

        # Auto-render on dropdown change
        self._dd_left.observe(self._on_compare, names='value')
        self._dd_right.observe(self._on_compare, names='value')

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _equity_for(self, label: str) -> tuple[str, object]:
        idx = self._labels().index(label)
        row = self.results.iloc[idx]
        return row['ticker'], row['strategy'], dict(row['params']), row['_result']

    def _plot_single(self, ax_equity, ax_dd, result, title: str, color: str):
        equity = result.equity
        norm_equity = equity / equity.iloc[0] * self.initial_capital

        ax_equity.plot(norm_equity, color=color, linewidth=1.5)
        ax_equity.set_title(title, fontsize=10)
        ax_equity.set_ylabel('Value ($)')
        ax_equity.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
        )

        # Trade markers
        if not result.trades.empty:
            buys  = result.trades[result.trades['action'] == 'BUY']
            sells = result.trades[result.trades['action'] == 'SELL']
            buy_eq  = norm_equity.reindex(buys.index,  method='nearest')
            sell_eq = norm_equity.reindex(sells.index, method='nearest')
            ax_equity.scatter(buy_eq.index,  buy_eq.values,  marker='^', color='green', s=40, zorder=5)
            ax_equity.scatter(sell_eq.index, sell_eq.values, marker='v', color='red',   s=40, zorder=5)

        # Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        ax_dd.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.35)
        ax_dd.set_ylabel('Drawdown')
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax_dd.set_xlabel('Date')

    def _on_compare(self, _=None):
        self._out.clear_output(wait=True)
        with self._out:
            ticker_l, strat_l, params_l, result_l = self._equity_for(self._dd_left.value)
            ticker_r, strat_r, params_r, result_r = self._equity_for(self._dd_right.value)

            param_str_l = ', '.join(f'{k}={v}' for k, v in params_l.items())
            param_str_r = ', '.join(f'{k}={v}' for k, v in params_r.items())
            title_l = f'{ticker_l} | {strat_l}\n({param_str_l})'
            title_r = f'{ticker_r} | {strat_r}\n({param_str_r})'

            fig = plt.figure(figsize=(15, 7))
            gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.3)

            ax_eq_l  = fig.add_subplot(gs[0, 0])
            ax_eq_r  = fig.add_subplot(gs[0, 1])
            ax_dd_l  = fig.add_subplot(gs[1, 0], sharex=ax_eq_l)
            ax_dd_r  = fig.add_subplot(gs[1, 1], sharex=ax_eq_r)

            self._plot_single(ax_eq_l, ax_dd_l, result_l, title_l, color='steelblue')
            self._plot_single(ax_eq_r, ax_dd_r, result_r, title_r, color='darkorange')

            plt.setp(ax_eq_l.get_xticklabels(), visible=False)
            plt.setp(ax_eq_r.get_xticklabels(), visible=False)

            # Summary stats table below the charts
            rows = self.results
            idx_l = self._labels().index(self._dd_left.value)
            idx_r = self._labels().index(self._dd_right.value)

            stats_cols = ['sharpe', 'total_return', 'max_drawdown', 'cagr', 'n_trades']
            stats = pd.DataFrame({
                f'{ticker_l}/{strat_l}': rows.iloc[idx_l][stats_cols],
                f'{ticker_r}/{strat_r}': rows.iloc[idx_r][stats_cols],
            })

            fig.text(0.5, -0.02, stats.to_string(), ha='center', fontsize=9,
                     fontfamily='monospace', va='top')

            plt.suptitle('Side-by-Side Strategy Comparison', fontsize=12, y=1.01)
            plt.tight_layout()
            plt.show()
            print()
            print(stats.to_string())

    # ------------------------------------------------------------------
    # Overlay view — all top-N equity curves on one chart
    # ------------------------------------------------------------------

    def plot_all(self, top_n: int = 10):
        """Plot the top N equity curves overlaid on a single chart."""
        fig, ax = plt.subplots(figsize=(13, 6))
        cmap = plt.cm.tab10

        for i, (_, row) in enumerate(self.results.head(top_n).iterrows()):
            equity = row['_result'].equity
            norm = equity / equity.iloc[0] * self.initial_capital
            label = f"#{i+1} {row['ticker']} {row['strategy']} (S={row['sharpe']:.2f})"
            ax.plot(norm, label=label, color=cmap(i / top_n), linewidth=1.2, alpha=0.8)

        ax.set_title(f'Top {top_n} Strategies — Normalised Equity Curves')
        ax.set_ylabel('Portfolio Value ($)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.legend(fontsize=8, loc='upper left')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------

    def show(self):
        """Display the interactive browser widget."""
        header = widgets.HTML('<b>MutationEngine Results Browser</b>')
        controls = widgets.HBox([self._dd_left, self._dd_right, self._btn])
        display(widgets.VBox([header, controls, self._out]))
        self._on_compare()  # render immediately
