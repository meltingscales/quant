# Skills Needed for Quantitative Trading

Based on *Quantitative Trading* by Ernest Chan and tailored to your background
as a senior security engineer with 8 years of Python and a CS/IT foundation.

---

## What You Already Have (Leverage These)

| Skill | Your Level | How It Applies |
|---|---|---|
| Python | 8 years | Primary language for all strategies, backtesting, and data analysis |
| SQL / MySQL | 6 years | Querying price/fundamental databases, tick data storage |
| Linux | 7 years | Running live trading systems, cron jobs, servers |
| Data pipelines | Strong | Building ingestion pipelines for market data feeds |
| Software design | 8 years | Writing clean, testable strategy code |
| Security mindset | Expert | Risk management, adversarial thinking about market edge cases |
| PowerBI / Python automation | Used at work | Executive metrics → portfolio performance dashboards |

---

## Mathematics

### Calculus (Priority: Medium)
You likely covered this in undergrad. Review focus areas:
- [ ] Derivatives and partial derivatives (used in options Greeks: delta, gamma, theta)
- [ ] Optimization (gradient descent for parameter fitting)
- [ ] Taylor series (Black-Scholes approximations)

> **Notebook:** `lessons/math/01_calculus_review.ipynb`

### Linear Algebra (Priority: High)
Critical for portfolio optimization and factor models.
- [ ] Vectors and matrices (covariance matrices, portfolio weights)
- [ ] Eigenvalues/eigenvectors (PCA for factor analysis)
- [ ] Matrix decomposition (Cholesky for correlated simulations)

> **Notebook:** `lessons/math/02_linear_algebra.ipynb`

### Statistics & Probability (Priority: High — start here)
Your most important foundation. You have general exposure but need finance-specific depth.
- [ ] Distributions: normal, log-normal, fat tails (asset returns are NOT normal)
- [ ] Hypothesis testing and p-values (avoiding overfitting/data snooping)
- [ ] Regression: OLS, ridge, LASSO (factor models)
- [ ] Time series: stationarity, autocorrelation, ADF test
- [ ] Cointegration (pairs trading foundation)
- [ ] Bayesian inference (position sizing, parameter updating)

> **Notebooks:** `lessons/stats/01_distributions.ipynb`, `02_hypothesis_testing.ipynb`, `03_time_series.ipynb`

---

## Financial Domain Knowledge (Priority: High — you have none yet)

### Markets & Instruments
- [ ] Asset classes: equities, futures, FX, options, crypto
- [ ] Order types: market, limit, stop, MOC, VWAP
- [ ] Market microstructure: bid-ask spread, order book, slippage
- [ ] Exchange mechanics: settlement, margin, short selling

> **Notebook:** `lessons/finance/01_markets_and_instruments.ipynb`

### Financial Mathematics
- [ ] Time value of money, discounting, NPV
- [ ] Return calculations: simple vs. log returns (use log returns)
- [ ] Risk metrics: Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio
- [ ] Value at Risk (VaR) and Expected Shortfall (CVaR)
- [ ] Options pricing basics: Black-Scholes, put-call parity, Greeks

> **Notebooks:** `lessons/finance/02_return_math.ipynb`, `03_risk_metrics.ipynb`, `04_options_basics.ipynb`

### Portfolio Theory
- [ ] Mean-variance optimization (Markowitz)
- [ ] Efficient frontier and capital market line
- [ ] CAPM and factor models (Fama-French 3/5-factor)
- [ ] Kelly criterion for position sizing

> **Notebook:** `lessons/finance/05_portfolio_theory.ipynb`

---

## Python Ecosystem for Quant (Priority: High)

You know Python well — you just need the finance-specific libraries.

| Library | Purpose | Priority |
|---|---|---|
| `pandas` | Time series manipulation, OHLCV data | High |
| `numpy` | Matrix math, vectorized operations | High |
| `scipy` | Statistical tests, optimization | High |
| `matplotlib` / `seaborn` | Charting equity curves, distributions | High |
| `statsmodels` | OLS regression, ADF test, cointegration | High |
| `yfinance` / `alpaca-py` | Free market data | Medium |
| `zipline-reloaded` / `backtrader` | Backtesting frameworks | Medium |
| `cvxpy` | Portfolio optimization (convex) | Medium |
| `scikit-learn` | ML models for alpha signals | Medium |
| `QuantLib` | Derivatives pricing | Low (later) |

> **Notebook:** `lessons/tools/01_quant_python_stack.ipynb`

---

## Strategy Development (The Core of Chan's Book)

### Mean Reversion
- [ ] Augmented Dickey-Fuller (ADF) test for stationarity
- [ ] Pairs trading and spread construction
- [ ] Ornstein-Uhlenbeck process (half-life of mean reversion)
- [ ] Bollinger Band strategies

> **Notebook:** `lessons/strategies/01_mean_reversion.ipynb`

### Momentum / Trend Following
- [ ] Time series momentum (TSMOM)
- [ ] Cross-sectional momentum (relative strength)
- [ ] Moving average crossovers, ATR-based sizing
- [ ] Trend filters: ADX, Hurst exponent

> **Notebook:** `lessons/strategies/02_momentum.ipynb`

### Backtesting Rigor (Priority: Critical — your security mindset helps here)
This is where most beginners blow up. Treat backtesting like threat modeling.
- [ ] Look-ahead bias (the silent killer)
- [ ] Survivorship bias (delisted stocks)
- [ ] Transaction costs and slippage modeling
- [ ] Walk-forward validation vs. in-sample/out-of-sample split
- [ ] Multiple comparisons problem (data snooping)
- [ ] Deflated Sharpe Ratio (Bailey & Lopez de Prado)

> **Notebook:** `lessons/backtesting/01_backtesting_pitfalls.ipynb`

---

## Execution & Infrastructure (Later Stage)

Once you have validated strategies:
- [ ] Broker APIs: Alpaca (free, good for learning), Interactive Brokers
- [ ] Paper trading before live
- [ ] Execution algorithms: TWAP, VWAP, implementation shortfall
- [ ] Live system architecture: data → signal → order → fill → reconcile
- [ ] Monitoring and alerting (you already know this from your security work)

> **Notebook:** `lessons/execution/01_broker_apis.ipynb`

---

## Recommended Learning Order

```
1. stats/01_distributions          ← Start here (2–3 sessions)
2. stats/02_hypothesis_testing
3. stats/03_time_series
4. finance/01_markets_and_instruments
5. finance/02_return_math
6. finance/03_risk_metrics
7. tools/01_quant_python_stack
8. math/02_linear_algebra
9. finance/05_portfolio_theory
10. strategies/01_mean_reversion    ← First real strategy
11. backtesting/01_backtesting_pitfalls
12. strategies/02_momentum
13. finance/04_options_basics       ← Optional unless trading options
14. execution/01_broker_apis        ← Only after strategy validation
```

---

## Resources

| Resource | Type | Notes |
|---|---|---|
| *Quantitative Trading* — Ernest Chan | Book | Primary text — work every example in a notebook |
| *Algorithmic Trading* — Ernest Chan | Book | Follow-up, more strategies |
| *Advances in Financial ML* — Lopez de Prado | Book | Advanced, read after Chan |
| QuantLib docs | Docs | Derivatives pricing reference |
| Alpaca Markets | Broker/API | Free paper trading, good for learning |
| Kaggle financial datasets | Data | Free OHLCV data for practice |
