# Bond Fund-of-Funds Monitor — Project Context

> Context file for the AI coding agent. Read this before every task.

## Purpose
A portfolio project supporting an application to a junior asset manager internship at a Geneva asset manager
running fixed income fund-of-funds and alternative strategies.
It shows that the author can do the three core tasks of the role:
1. **Portfolio monitoring, risk analysis and performance evaluation** of a fixed income portfolio.
2. **Reporting**: an automated one-page weekly factsheet.
3. **Workflow automation**: a Python pipeline plus an Excel/VBA dashboard.

The fictional fund is a **fund of bond funds**: ~8 bond ETFs act as proxies for the underlying funds.

## Hard rules
- **Fictional portfolio only.** Weights are invented; no real client or real fund holdings.
- **No branding** of any real asset manager or bank. The app is called "Bond Fund-of-Funds Monitor", an independent student prototype.
- Visible disclaimer on every page and on the factsheet: *"Student prototype — fictional portfolio — not investment advice."*
- ETFs are **proxies** (US-listed, not UCITS). Say so in the UI and the README.
- No secrets in the repo. FRED and yfinance need no API key.
- The public demo must work offline from cached CSVs if yfinance or FRED fail.

## Tech stack
- Python 3.11+, Streamlit (multipage), pandas, numpy, statsmodels, plotly, yfinance
- Rates and spreads from FRED (direct CSV download, no key)
- Reporting: matplotlib + reportlab (PDF), openpyxl (Excel export)
- Excel dashboard: a VBA module (`excel/Dashboard.bas`) imported into an `.xlsm` template
- Tests: pytest for every function in `core/`
- Deployment: Streamlit Community Cloud

## Portfolio (base currency USD, weekly data)
| Sleeve | Ticker | Weight |
|---|---|---|
| Short Treasuries | SHY | 15% |
| Intermediate Treasuries | IEF | 20% |
| Long Treasuries | TLT | 10% |
| Inflation-linked | TIP | 10% |
| Investment grade credit | LQD | 15% |
| High yield credit | HYG | 10% |
| Emerging markets USD | EMB | 10% |
| International (USD-hedged) | BNDX | 10% |
Equity benchmark for correlations: SPY.

## Structure
```
app.py                       # landing page: problem, solution, modules, disclaimer
pages/
  1_Overview.py
  2_Rates_and_Credit.py
  3_Stress_Tests.py
  4_Factsheet.py
core/
  data.py                    # prices (yfinance), FRED series, cache, CSV fallback
  portfolio.py               # weights, NAV, returns, rebalancing
  fixed_income.py            # duration, DV01, spread duration, empirical duration
  risk.py                    # vol, VaR, ES, drawdown, Sharpe, Sortino, risk contributions
  stress.py                  # rate/spread shocks, 2022 historical replay
  limits.py                  # limits and alerts (info / warning / breach)
  reporting.py               # PDF factsheet + Excel export
data/
  etf_characteristics.csv    # duration, spread duration, convexity, hedged flag, as-of date
  cache/                     # cached prices and FRED series
excel/
  Dashboard.bas              # VBA module
  README_excel.md            # how to import the module and run the macro
tests/
.streamlit/config.toml
requirements.txt
README.md
```

## Coding standards
- Type hints, small pure functions in `core/`, UI only in `pages/`.
- Every financial formula documented in its docstring: definition, assumptions, limits.
- Weekly frequency (Friday close) for returns and regressions.
- Durations are **as-of dated** inputs; the UI shows the as-of date.
- English in code and UI.

## Author's learning goal
The author must be able to explain every formula and design choice in an interview.
After each task, briefly explain **what** was built, **why**, and **which assumptions** were made.
