import numpy as np
import pandas as pd
import pytest

from core import portfolio as pf


def make_returns(a, b, start="2024-01-19"):
    idx = pd.date_range(start, periods=len(a), freq="W-FRI")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


# --- weights -----------------------------------------------------------------

def test_target_weights_match_project_spec():
    assert list(pf.TARGET_WEIGHTS) == ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "BNDX"]
    assert pf.TARGET_WEIGHTS["IEF"] == 0.20
    w = pf.validate_weights(pf.TARGET_WEIGHTS)
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("weights, match", [
    ({"A": 0.6, "B": 0.6}, "sum to 1"),
    ({"A": 1.2, "B": -0.2}, ">= 0"),
    ({"A": float("nan"), "B": 1.0}, "finite"),
    ({}, "empty"),
])
def test_validate_weights_rejects_invalid(weights, match):
    with pytest.raises(ValueError, match=match):
        pf.validate_weights(weights)


# --- weekly_returns ----------------------------------------------------------

def test_weekly_returns_are_simple_returns():
    idx = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=idx)
    r = pf.weekly_returns(prices)
    assert list(r.index) == list(idx[1:])
    assert r["A"].tolist() == pytest.approx([0.10, -0.10])


def test_weekly_returns_rejects_non_positive_prices():
    idx = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
    with pytest.raises(ValueError, match="positive"):
        pf.weekly_returns(pd.DataFrame({"A": [100.0, 0.0]}, index=idx))


# --- portfolio_returns -------------------------------------------------------

def test_portfolio_returns_drift_then_monthly_reset():
    # Weeks: Jan 19, Jan 26 (last Friday of January -> rebalance), Feb 2, Feb 9.
    returns = make_returns(a=[0.10, 0.10, 0.10, 0.0], b=[0.0, 0.0, 0.0, 0.0])
    weights = {"A": 0.5, "B": 0.5}

    rp = pf.portfolio_returns(returns, weights, rebalance="M")

    w_a_week2 = 0.5 * 1.10 / 1.05  # drifted weight of A after week 1
    assert rp.iloc[0] == pytest.approx(0.05)
    assert rp.iloc[1] == pytest.approx(w_a_week2 * 0.10)
    assert rp.iloc[2] == pytest.approx(0.05)  # back to 50/50 after the January rebalance
    assert rp.iloc[3] == pytest.approx(0.0)


def test_buy_and_hold_matches_weighted_asset_growth():
    rng = np.random.default_rng(0)
    returns = make_returns(rng.normal(0, 0.01, 30), rng.normal(0, 0.02, 30))
    weights = {"A": 0.3, "B": 0.7}

    rp = pf.portfolio_returns(returns, weights, rebalance=None)

    growth = (1 + returns).cumprod()
    expected_nav = 0.3 * growth["A"] + 0.7 * growth["B"]
    assert pf.nav(rp, base=1.0).to_numpy() == pytest.approx(expected_nav.to_numpy())


def test_monthly_rebalancing_differs_from_buy_and_hold():
    rng = np.random.default_rng(1)
    returns = make_returns(rng.normal(0, 0.01, 20), rng.normal(0, 0.02, 20))
    weights = {"A": 0.5, "B": 0.5}
    monthly = pf.portfolio_returns(returns, weights, "M")
    hold = pf.portfolio_returns(returns, weights, None)
    first_rebalance = 1  # Jan 26 is the last week of January
    assert monthly.iloc[: first_rebalance + 1].to_numpy() == pytest.approx(hold.iloc[: first_rebalance + 1].to_numpy())
    assert not np.allclose(monthly.iloc[first_rebalance + 1:], hold.iloc[first_rebalance + 1:])


def test_identical_asset_returns_give_the_same_portfolio_return():
    returns = make_returns([0.01, -0.02, 0.03], [0.01, -0.02, 0.03])
    rp = pf.portfolio_returns(returns, {"A": 0.4, "B": 0.6})
    assert rp.tolist() == pytest.approx([0.01, -0.02, 0.03])


def test_portfolio_returns_ignores_extra_columns_and_uses_default_weights():
    idx = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    returns = pd.DataFrame(0.01, index=idx, columns=list(pf.TARGET_WEIGHTS) + ["SPY"])
    returns["SPY"] = 0.5
    rp = pf.portfolio_returns(returns)
    assert rp.tolist() == pytest.approx([0.01] * 4)


def test_portfolio_returns_input_errors():
    returns = make_returns([0.01, np.nan], [0.0, 0.0])
    with pytest.raises(ValueError, match="missing values"):
        pf.portfolio_returns(returns, {"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError, match="lack columns.*C"):
        pf.portfolio_returns(returns, {"A": 0.5, "C": 0.5})
    with pytest.raises(ValueError, match="rebalance"):
        pf.portfolio_returns(returns.fillna(0), {"A": 0.5, "B": 0.5}, rebalance="W")


# --- nav ---------------------------------------------------------------------

def test_nav_compounds_from_base():
    r = pd.Series([0.10, -0.10, 0.05])
    assert pf.nav(r).tolist() == pytest.approx([110.0, 99.0, 103.95])
    assert pf.nav(r, base=1.0).iloc[-1] == pytest.approx(1.0395)


def test_nav_rejects_missing_returns():
    with pytest.raises(ValueError):
        pf.nav(pd.Series([0.01, np.nan]))
