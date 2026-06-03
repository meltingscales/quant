# Repo Layout

## Root

- `README.md` - project overview and book list
- `SKILLS-NEEDED.md` - learning roadmap tailored to Henry's background
- `pyproject.toml` - Python dependencies managed by uv
- `uv.lock` - locked dependency versions (committed)
- `.python-version` - Python version pin for uv
- `justfile` - common commands (`just lab`, `just run <notebook>`, `just add <package>`)
- `.gitignore` - excludes `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`

## `.claude/`

Claude AI configuration for this project.

- `CLAUDE.md` - top-level instructions for Claude
- `rules/` - scoped rule files loaded by Claude
  - `my-skills.md` - Henry's full resume/background (refreshable from drakonix.systems/pages/resume)
  - `repo-layout-files.md` - this file
- `settings.local.json` - local Claude Code settings

## `lessons/`

Jupyter notebooks (`.ipynb`) for structured learning. Managed via `uv run jupyter lab`.

Naming convention: `NN_topic_name.ipynb` (zero-padded number prefix for ordering).

### Topic subdirectories

- `lessons/stats/` - probability and statistics (distributions, hypothesis testing, time series)
- `lessons/math/` - linear algebra, calculus review
- `lessons/finance/` - financial domain knowledge (markets, return math, risk metrics, options, portfolio theory)
- `lessons/tools/` - Python quant stack (pandas, numpy, scipy, statsmodels, etc.)
- `lessons/strategies/` - trading strategies (mean reversion, momentum)
- `lessons/backtesting/` - backtesting methodology and pitfalls
- `lessons/execution/` - broker APIs, live trading infrastructure

### Homework

Each topic directory may contain:

- `lessons/TOPIC/homework/` - exercises and scratchpad notebooks
- `lessons/TOPIC/homework/solutions/` - worked solutions

## `DrakonixBacktester/`

Custom backtesting framework written from scratch in Python. Used to validate strategies developed in the lesson notebooks.
