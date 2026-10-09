import numpy as np
import pandas as pd
import pytest

from core import data

TICKERS = ["AAA", "BBB"]


def fake_yf_frame(tickers, start="2024-01-01", end="2024-02-29"):
    """Daily yfinance-like frame with (Price, Ticker) MultiIndex columns."""
    dates = pd.bdate_range(start, end)
    columns = pd.MultiIndex.from_product([["Close", "Open"], tickers], names=["Price", "Ticker"])
    values = np.tile(np.arange(1, len(dates) + 1, dtype=float)[:, None], (1, len(columns)))
    return pd.DataFrame(values, index=dates, columns=columns)


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


FRED_CSV = (
    "observation_date,DGS10\n"
    "2024-01-01,.\n"
    "2024-01-02,4.00\n"
    "2024-01-03,4.10\n"
    "2024-01-04,4.20\n"
    "2024-01-05,4.30\n"
    "2024-01-08,4.40\n"
    "2024-01-11,4.50\n"
    "2024-01-12,.\n"
)


# --- to_weekly ---------------------------------------------------------------

def test_to_weekly_uses_last_value_and_friday_labels():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-08"]))
    weekly = data.to_weekly(s)
    assert list(weekly.index) == list(pd.to_datetime(["2024-01-05", "2024-01-12"]))
    assert all(weekly.index.dayofweek == 4)
    assert weekly.tolist() == [2.0, 3.0]


def test_to_weekly_keeps_empty_weeks_as_nan():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-05", "2024-01-19"]))
    weekly = data.to_weekly(s)
    assert pd.isna(weekly.loc["2024-01-12"])


# Daily data up to Thursday 2026-10-08; the last weekly label is Friday 2026-10-09.
LAST_WEEK = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    index=pd.to_datetime(["2026-09-30", "2026-10-02", "2026-10-06", "2026-10-07", "2026-10-08"]),
)


@pytest.mark.parametrize("now, kept", [
    (pd.Timestamp("2026-10-07 12:00", tz="America/New_York"), False),  # Wednesday
    (pd.Timestamp("2026-10-09 15:59", tz="America/New_York"), False),  # Friday before close
    (pd.Timestamp("2026-10-09 14:35", tz="Europe/Zurich"), False),     # = 08:35 New York
    (pd.Timestamp("2026-10-09 16:00", tz="America/New_York"), True),   # at the close
    (pd.Timestamp("2026-10-09 22:30", tz="Europe/Zurich"), True),      # = 16:30 New York
    (pd.Timestamp("2026-10-10 09:00"), True),                          # naive = New York
])
def test_to_weekly_drops_unfinished_last_week(now, kept):
    weekly = data.to_weekly(LAST_WEEK, now=now)
    expected_last = pd.Timestamp("2026-10-09" if kept else "2026-10-02")
    assert weekly.index[-1] == expected_last
    assert weekly.loc["2026-10-02"] == 2.0


def test_to_weekly_can_keep_unfinished_week():
    now = pd.Timestamp("2026-10-07 12:00", tz="America/New_York")
    weekly = data.to_weekly(LAST_WEEK, drop_incomplete=False, now=now)
    assert weekly.index[-1] == pd.Timestamp("2026-10-09")
    assert weekly.iloc[-1] == 5.0


# --- get_prices --------------------------------------------------------------

def test_get_prices_resamples_to_friday_and_writes_cache(monkeypatch, tmp_path):
    raw = fake_yf_frame(TICKERS)
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: raw)

    prices = data.get_prices(TICKERS, start="2024-01-01", cache_dir=tmp_path)

    assert list(prices.columns) == TICKERS
    assert all(prices.index.dayofweek == 4)
    friday = pd.Timestamp("2024-01-12")
    assert prices.loc[friday, "AAA"] == raw.loc[friday, ("Close", "AAA")]
    assert (tmp_path / "prices.csv").exists()


def test_get_prices_uses_thursday_when_friday_is_missing(monkeypatch, tmp_path):
    raw = fake_yf_frame(TICKERS).drop(pd.Timestamp("2024-01-12"))
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: raw)

    prices = data.get_prices(TICKERS, cache_dir=tmp_path)

    assert prices.loc["2024-01-12", "AAA"] == raw.loc["2024-01-11", ("Close", "AAA")]


def test_get_prices_falls_back_to_cache_on_error(monkeypatch, tmp_path):
    raw = fake_yf_frame(TICKERS)
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: raw)
    fresh = data.get_prices(TICKERS, start="2024-01-01", cache_dir=tmp_path)

    def broken(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(data.yf, "download", broken)
    with pytest.warns(UserWarning, match="cached prices"):
        cached = data.get_prices(TICKERS, start="2024-01-01", cache_dir=tmp_path)

    pd.testing.assert_frame_equal(cached, fresh, check_freq=False)


def test_get_prices_falls_back_when_a_ticker_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: fake_yf_frame(TICKERS))
    data.get_prices(TICKERS, cache_dir=tmp_path)

    incomplete = fake_yf_frame(TICKERS)
    incomplete[("Close", "BBB")] = np.nan
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: incomplete)
    with pytest.warns(UserWarning, match="BBB"):
        prices = data.get_prices(TICKERS, cache_dir=tmp_path)
    assert prices["BBB"].notna().all()


def test_get_prices_raises_without_cache(tmp_path):
    with pytest.warns(UserWarning):
        with pytest.raises(RuntimeError, match="no cache"):
            data.get_prices(TICKERS, cache_dir=tmp_path)


def test_get_prices_drops_unfinished_week(monkeypatch, tmp_path):
    raw = fake_yf_frame(TICKERS, start="2026-09-28", end="2026-10-08")
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: raw)
    monkeypatch.setattr(data, "_now", lambda: pd.Timestamp("2026-10-09 08:35", tz="America/New_York"))

    prices = data.get_prices(TICKERS, start="2026-09-28", cache_dir=tmp_path)

    assert prices.index[-1] == pd.Timestamp("2026-10-02")
    cached = pd.read_csv(tmp_path / "prices.csv", index_col=0, parse_dates=True)
    assert cached.index[-1] == pd.Timestamp("2026-10-02")


def test_get_prices_default_tickers_include_portfolio_and_spy(monkeypatch, tmp_path):
    seen = {}

    def fake_download(tickers, **kwargs):
        seen["tickers"] = tickers
        return fake_yf_frame(tickers)

    monkeypatch.setattr(data.yf, "download", fake_download)
    prices = data.get_prices(cache_dir=tmp_path)
    assert seen["tickers"] == data.PORTFOLIO_TICKERS + ["SPY"]
    assert list(prices.columns) == seen["tickers"]


# --- get_fred_series ---------------------------------------------------------

def test_get_fred_series_parses_missing_converts_to_decimal_and_resamples(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return FakeResponse(FRED_CSV)

    monkeypatch.setattr(data.requests, "get", fake_get)
    s = data.get_fred_series("DGS10", cache_dir=tmp_path)

    assert calls == ["https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"]
    assert s.name == "DGS10"
    assert list(s.index) == list(pd.to_datetime(["2024-01-05", "2024-01-12"]))
    assert s.loc["2024-01-05"] == pytest.approx(0.0430)
    assert s.loc["2024-01-12"] == pytest.approx(0.0450)  # Friday is "." -> Thursday value
    assert (tmp_path / "DGS10.csv").exists()


def test_get_fred_series_drops_unfinished_week(monkeypatch, tmp_path):
    monkeypatch.setattr(data.requests, "get", lambda url, timeout: FakeResponse(FRED_CSV))
    monkeypatch.setattr(data, "_now", lambda: pd.Timestamp("2024-01-12 10:00", tz="America/New_York"))

    s = data.get_fred_series("DGS10", cache_dir=tmp_path)

    assert list(s.index) == [pd.Timestamp("2024-01-05")]


def test_baa10y_is_a_fred_series_in_the_coverage_report(monkeypatch):
    assert "BAA10Y" in data.FRED_SERIES
    idx = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    monkeypatch.setattr(data, "get_prices", lambda: pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx))
    monkeypatch.setattr(data, "get_fred_series", lambda sid: pd.Series([0.02, 0.021, 0.022], index=idx, name=sid))

    report = data.data_coverage_report()

    assert report.loc["BAA10Y", "source"] == "FRED"
    assert set(data.FRED_SERIES) <= set(report.index)


def test_get_fred_series_falls_back_to_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(data.requests, "get", lambda url, timeout: FakeResponse(FRED_CSV))
    fresh = data.get_fred_series("DGS10", cache_dir=tmp_path)

    monkeypatch.setattr(data.requests, "get", lambda url, timeout: FakeResponse("", status_code=503))
    with pytest.warns(UserWarning, match="DGS10"):
        cached = data.get_fred_series("DGS10", cache_dir=tmp_path)

    pd.testing.assert_series_equal(cached, fresh, check_freq=False)


def test_get_fred_series_rejects_unexpected_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(data.requests, "get", lambda url, timeout: FakeResponse("<html>error</html>"))
    with pytest.warns(UserWarning):
        with pytest.raises(RuntimeError, match="no cache"):
            data.get_fred_series("DGS10", cache_dir=tmp_path)


# --- load_etf_characteristics --------------------------------------------------

HEADER = "ticker,sleeve,duration,spread_duration,convexity,hedged,as_of\n"


def write_csv(tmp_path, body, header=HEADER):
    path = tmp_path / "etf.csv"
    path.write_text(header + body)
    return path


def test_load_etf_characteristics_parses_types(tmp_path):
    path = write_csv(tmp_path, "AAA,Sleeve A,1.9,0.0,0.05,false,2026-09-30\n"
                               "BBB,Sleeve B,6.5,6.4,0.6,TRUE,2026-09-30\n")
    df = data.load_etf_characteristics(path)
    assert list(df.index) == ["AAA", "BBB"]
    assert df.loc["BBB", "duration"] == 6.5
    assert df["hedged"].tolist() == [False, True]
    assert df.loc["AAA", "as_of"] == pd.Timestamp("2026-09-30")


def test_load_etf_characteristics_missing_column(tmp_path):
    path = write_csv(tmp_path, "AAA,Sleeve A,1.9,0.0,false,2026-09-30\n",
                     header="ticker,sleeve,duration,spread_duration,hedged,as_of\n")
    with pytest.raises(ValueError, match="missing column.*convexity"):
        data.load_etf_characteristics(path)


def test_load_etf_characteristics_missing_value_names_ticker_and_column(tmp_path):
    path = write_csv(tmp_path, "AAA,Sleeve A,1.9,0.0,0.05,false,2026-09-30\n"
                               "BBB,Sleeve B,,6.4,0.6,true,2026-09-30\n")
    with pytest.raises(ValueError, match="BBB: duration"):
        data.load_etf_characteristics(path)


def test_load_etf_characteristics_empty_convexity_is_nan_with_warning(tmp_path):
    path = write_csv(tmp_path, "AAA,Sleeve A,1.9,0.0,,false,2026-09-30\n"
                               "BBB,Sleeve B,6.5,6.4,0.6,true,2026-09-30\n")
    with pytest.warns(UserWarning, match="convexity is empty for AAA"):
        df = data.load_etf_characteristics(path)
    assert pd.isna(df.loc["AAA", "convexity"])
    assert df.loc["BBB", "convexity"] == 0.6
    assert df["convexity"].dtype == float


def test_load_etf_characteristics_rejects_bad_values(tmp_path):
    path = write_csv(tmp_path, "AAA,Sleeve A,abc,0.0,0.05,false,2026-09-30\n")
    with pytest.raises(ValueError, match="non-numeric duration"):
        data.load_etf_characteristics(path)
    path = write_csv(tmp_path, "AAA,Sleeve A,1.9,0.0,abc,false,2026-09-30\n")
    with pytest.raises(ValueError, match="non-numeric convexity"):
        data.load_etf_characteristics(path)
    path = write_csv(tmp_path, "AAA,Sleeve A,1.9,0.0,0.05,maybe,2026-09-30\n")
    with pytest.raises(ValueError, match="hedged"):
        data.load_etf_characteristics(path)
    path = write_csv(tmp_path, "AAA,Sleeve A,1.9,0.0,0.05,false,30/09/2026\n")
    with pytest.raises(ValueError, match="as_of"):
        data.load_etf_characteristics(path)


def test_etf_characteristics_template_lists_the_eight_portfolio_etfs():
    template = pd.read_csv(data.ETF_CHARACTERISTICS_PATH)
    assert list(template.columns) == data.ETF_REQUIRED_COLUMNS
    assert template["ticker"].tolist() == data.PORTFOLIO_TICKERS
    assert template["sleeve"].notna().all()


# --- data_coverage_report ------------------------------------------------------

def test_data_coverage_report_counts_obs_and_internal_gaps():
    idx = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    prices = pd.DataFrame({
        "AAA": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "NEW": [np.nan, np.nan, 3.0, 4.0, 5.0, np.nan],  # late inception, stops early
    }, index=idx)
    fred = {"DGS10": pd.Series([0.04, np.nan, 0.041], index=idx[:3])}

    report = data.data_coverage_report(prices=prices, fred=fred)

    assert list(report.index) == ["AAA", "NEW", "DGS10"]
    assert report.loc["AAA", "n_obs"] == 5 and report.loc["AAA", "n_missing"] == 1
    assert report.loc["NEW", "first_date"] == idx[2]
    assert report.loc["NEW", "last_date"] == idx[4]
    assert report.loc["NEW", "n_missing"] == 0
    assert report.loc["DGS10", "source"] == "FRED"
    assert report.loc["DGS10", "n_missing"] == 1
