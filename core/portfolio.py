"""Fictional portfolio: target weights, weekly returns, rebalanced portfolio returns and NAV."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

TARGET_WEIGHTS: dict[str, float] = {
    "SHY": 0.15,  # Short Treasuries
    "IEF": 0.20,  # Intermediate Treasuries
    "TLT": 0.10,  # Long Treasuries
    "TIP": 0.10,  # Inflation-linked
    "LQD": 0.15,  # Investment grade credit
    "HYG": 0.10,  # High yield credit
    "EMB": 0.10,  # Emerging markets USD
    "BNDX": 0.10,  # International (USD-hedged)
}

_REBALANCE_PERIODS = {"M": "M", "Q": "Q"}


def validate_weights(weights: Mapping[str, float], tol: float = 1e-8) -> pd.Series:
    """Check that weights describe a fully invested, long-only portfolio.

    Rules: every weight is finite and >= 0, and the weights sum to 1 (within ``tol``).

    Assumptions: no leverage, no short positions, no cash sleeve (cash would have to
    be an explicit line).

    Returns the weights as a float Series indexed by ticker.

    Raises:
        ValueError: if the weights are empty, non-finite, negative or do not sum to 1.
    """
    w = pd.Series(dict(weights), dtype=float)
    if w.empty:
        raise ValueError("weights are empty")
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError(f"weights must be finite numbers: {w.to_dict()}")
    negative = w[w < 0]
    if not negative.empty:
        raise ValueError(f"weights must be >= 0, got negative: {negative.to_dict()}")
    total = float(w.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"weights must sum to 1, got {total:.10f}")
    return w


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple weekly returns r_t = P_t / P_{t-1} - 1.

    Definition: simple (arithmetic) returns, not log returns, because simple returns
    aggregate across assets: the portfolio return is the weighted sum of asset returns.

    Assumptions: ``prices`` are weekly adjusted (total-return) prices, one column per
    ticker, sorted by date. The first row (no previous price) is dropped.

    Limits: no forward-fill; a missing price yields NaN for that week and the next
    (the gap is not silently bridged). Prices must be strictly positive.
    """
    prices = prices.sort_index()
    if (prices <= 0).to_numpy().any():
        raise ValueError("prices must be strictly positive")
    return prices.pct_change(fill_method=None).iloc[1:]


def portfolio_returns(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    rebalance: str | None = "M",
) -> pd.Series:
    """Portfolio returns with periodic rebalancing to target weights and drift in between.

    Definition: with weights w_{t-1} held at the start of week t,

        r_p,t   = sum_i w_{i,t-1} * r_{i,t}
        w_{i,t} = w_{i,t-1} * (1 + r_{i,t}) / (1 + r_p,t)        (drift)

    Each asset's weight grows with its own return relative to the portfolio. On the
    last weekly observation of each calendar month (``rebalance="M"``; "Q" for
    quarterly), after that week's return, weights are reset to the targets. The
    first week starts at the target weights. ``rebalance=None`` is buy-and-hold.

    Assumptions: no transaction costs, taxes or cash drag; rebalancing happens at the
    Friday close used for pricing; ETFs proxy the underlying funds.

    Limits: with weekly data, "month-end" is the last Friday of the month, not the
    actual last business day, so a few days per month are approximated. Columns of
    ``returns`` not in ``weights`` (e.g. SPY) are ignored.

    Raises:
        ValueError: on invalid weights, unknown ``rebalance``, tickers missing from
            ``returns`` or missing returns for a held asset.
    """
    w_target = validate_weights(weights if weights is not None else TARGET_WEIGHTS)
    if rebalance is not None and rebalance not in _REBALANCE_PERIODS:
        raise ValueError(f"rebalance must be one of {sorted(_REBALANCE_PERIODS)} or None, got {rebalance!r}")

    missing = [t for t in w_target.index if t not in returns.columns]
    if missing:
        raise ValueError(f"returns lack columns for: {', '.join(missing)}")

    r = returns[w_target.index].sort_index()
    if r.isna().to_numpy().any():
        first_nan = r.index[r.isna().any(axis=1)][0]
        raise ValueError(f"returns contain missing values (first at {first_nan.date()}); align data first")

    if rebalance is None:
        rebalance_flags = np.zeros(len(r), dtype=bool)
    else:
        periods = r.index.to_period(_REBALANCE_PERIODS[rebalance])
        rebalance_flags = np.append(periods[1:] != periods[:-1], False)

    target = w_target.to_numpy()
    weights_t = target.copy()
    values = r.to_numpy()
    out = np.empty(len(r))
    for t in range(len(r)):
        port_r = float(weights_t @ values[t])
        out[t] = port_r
        weights_t = weights_t * (1.0 + values[t]) / (1.0 + port_r)
        if rebalance_flags[t]:
            weights_t = target.copy()

    return pd.Series(out, index=r.index, name="portfolio")


def nav(portfolio_returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Net asset value index: NAV_t = base * prod_{s<=t} (1 + r_s).

    Definition: compounded growth of ``base`` invested at the start of the first return
    period. The first value is base * (1 + r_1), i.e. the NAV at the end of week 1.

    Assumptions: returns are simple and gross of any fund-of-funds management fee
    (not modelled); ETF expense ratios are already reflected in the prices.

    Limits: missing returns are not allowed (they would break the compounding).
    """
    if portfolio_returns.isna().any():
        raise ValueError("portfolio returns contain missing values")
    return (base * (1.0 + portfolio_returns).cumprod()).rename("nav")
