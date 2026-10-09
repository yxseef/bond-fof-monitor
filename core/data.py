"""Data layer: ETF prices (yfinance), FRED rates and spreads, local CSV cache and offline fallback."""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
ETF_CHARACTERISTICS_PATH = DATA_DIR / "etf_characteristics.csv"

PORTFOLIO_TICKERS: list[str] = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "BNDX"]
BENCHMARK_TICKER = "SPY"
PRICE_TICKERS: list[str] = PORTFOLIO_TICKERS + [BENCHMARK_TICKER]

FRED_SERIES: dict[str, str] = {
    "DGS10": "10-Year Treasury constant maturity yield",
    "DGS2": "2-Year Treasury constant maturity yield",
    "BAMLC0A0CM": "ICE BofA US Corporate (IG) option-adjusted spread",
    "BAMLH0A0HYM2": "ICE BofA US High Yield option-adjusted spread",
    "BAA10Y": "Moody's Baa corporate yield minus 10-Year Treasury yield",
}
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

WEEKLY_FREQ = "W-FRI"
NEW_YORK_TZ = "America/New_York"
NEW_YORK_CLOSE = pd.Timedelta(hours=16)

ETF_REQUIRED_COLUMNS: list[str] = [
    "ticker",
    "sleeve",
    "duration",
    "spread_duration",
    "convexity",
    "hedged",
    "as_of",
]
_ETF_NUMERIC_COLUMNS = ["duration", "spread_duration", "convexity"]
_ETF_OPTIONAL_COLUMNS = ["convexity"]
_TRUE_VALUES = {"true", "1", "yes", "y"}
_FALSE_VALUES = {"false", "0", "no", "n"}


def _resolve_cache_dir(cache_dir: Path | str | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else CACHE_DIR


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NEW_YORK_TZ)


def to_weekly(
    data: pd.DataFrame | pd.Series,
    drop_incomplete: bool = True,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame | pd.Series:
    """Resample daily observations to weekly, labelled on Friday ("W-FRI").

    Definition: each weekly value is the last *available* observation of the week
    ending on Friday (Saturday to Friday). NaNs are skipped, so if Friday is a
    market holiday, Thursday's value is used and labelled with Friday's date.

    Incomplete week: a week is complete once its Friday 16:00 New York time (NYSE
    close) has passed. With ``drop_incomplete=True``, the last week is dropped if
    ``now`` is before that time, so it never holds a mid-week value labelled as a
    Friday close. ``now`` defaults to the current time; a naive ``now`` is read as
    New York time.

    Assumptions: the input index is a DatetimeIndex of daily (business-day) data.

    Limits: a week without any observation yields NaN (no forward-fill, so gaps
    stay visible). The 16:00 cut-off ignores early closes (e.g. day after
    Thanksgiving) and data-publication lags (FRED posts a day's value later).
    """
    weekly = data.sort_index().resample(WEEKLY_FREQ).last()
    if not drop_incomplete or weekly.empty:
        return weekly

    now = pd.Timestamp(now) if now is not None else _now()
    now = now.tz_localize(NEW_YORK_TZ) if now.tz is None else now.tz_convert(NEW_YORK_TZ)
    last_close = (weekly.index[-1].normalize() + NEW_YORK_CLOSE).tz_localize(NEW_YORK_TZ)
    if now < last_close:
        weekly = weekly.iloc[:-1]
    return weekly


def _download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Download daily adjusted closes from yfinance; raise if anything is missing."""
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if missing:
        raise RuntimeError(f"yfinance returned no prices for: {', '.join(missing)}")

    close = close[tickers].astype(float)
    close.columns = pd.Index(tickers)
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).tz_localize(None)
    close.index.name = "date"
    return close


def get_prices(
    tickers: Iterable[str] | None = None,
    start: str = "2015-01-01",
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Weekly adjusted prices (Friday close) for the portfolio ETFs and the SPY benchmark.

    Definition: daily closes adjusted for splits and distributions (yfinance
    ``auto_adjust=True``), so price changes are total returns (coupons reinvested).
    They are resampled to weekly with ``to_weekly`` (last close of each week, "W-FRI").

    Cache and fallback: on success, the weekly prices are written to
    ``data/cache/prices.csv``. If the download fails or is incomplete (yfinance often
    returns empty columns instead of raising), a ``UserWarning`` is emitted and the
    cache is read instead, filtered to ``tickers`` and ``start``.

    Assumptions: adjusted prices are a fair proxy of fund total return; dividends are
    reinvested at the ex-date close; prices are in USD.

    Limits: yfinance is an unofficial Yahoo Finance API (data can be revised or
    temporarily unavailable). The adjusted history changes after each new
    distribution, so cached and fresh prices can differ slightly in level (returns
    are unaffected). The cache only contains the tickers of the last successful download.

    Raises:
        RuntimeError: if the download fails and no usable cache exists.
    """
    tickers = list(tickers) if tickers is not None else list(PRICE_TICKERS)
    cache_path = _resolve_cache_dir(cache_dir) / "prices.csv"

    try:
        weekly = to_weekly(_download_prices(tickers, start))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_csv(cache_path)
        return weekly
    except Exception as exc:  # noqa: BLE001 - any download failure triggers the fallback
        warnings.warn(
            f"Price download failed ({exc!r}); using cached prices from {cache_path}.",
            UserWarning,
            stacklevel=2,
        )

    if not cache_path.exists():
        raise RuntimeError(f"Price download failed and no cache found at {cache_path}.")
    cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    missing = [t for t in tickers if t not in cached.columns]
    if missing:
        raise RuntimeError(f"Cached prices at {cache_path} lack tickers: {', '.join(missing)}.")
    cached.index.name = "date"
    return cached.loc[cached.index >= pd.Timestamp(start), tickers]


def _download_fred(series_id: str, timeout: float) -> pd.Series:
    """Download a FRED series as daily values in percent (missing values as NaN)."""
    response = requests.get(FRED_CSV_URL.format(series_id=series_id), timeout=timeout)
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.text), na_values=["."])
    if series_id not in raw.columns:
        raise RuntimeError(f"FRED response has no column {series_id!r}: {list(raw.columns)}")
    dates = pd.to_datetime(raw.iloc[:, 0], errors="raise")
    values = pd.to_numeric(raw[series_id], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates, name="date"), name=series_id)
    if series.dropna().empty:
        raise RuntimeError(f"FRED series {series_id} contains no numeric values")
    return series


def get_fred_series(
    series_id: str,
    cache_dir: Path | str | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Weekly FRED series (yield or spread) in decimal, e.g. 4.25% -> 0.0425.

    Definition: daily values from the FRED CSV endpoint (no API key). FRED marks
    missing days (holidays) with ".", which are read as NaN. Values are published in
    percent and divided by 100. The weekly value is the last available observation
    of the week ending Friday; an unfinished last week is dropped (see ``to_weekly``).

    Series used:
      - DGS10, DGS2: Treasury constant-maturity yields.
      - BAMLC0A0CM, BAMLH0A0HYM2: ICE BofA US IG and HY option-adjusted spreads vs
        Treasuries. Because of ICE licensing, FRED only serves about the last 3 years.
      - BAA10Y: Moody's Baa corporate yield minus the 10-year Treasury yield. Used as
        a long-history IG credit spread proxy (e.g. for the 2022 replay). It is a
        yield difference (no option adjustment, Baa only, maturity mismatch), so its
        level is not directly comparable with the ICE OAS.

    Cache and fallback: on success the weekly series is written to
    ``data/cache/<ID>.csv``; on failure a ``UserWarning`` is emitted and the cache is read.

    Assumptions: the Friday value represents the whole week (end-of-week snapshot,
    consistent with Friday-close prices).

    Limits: FRED restricts the history of third-party (ICE) series; constant-maturity
    yields are interpolated par yields, not traded instruments; Treasury yields close
    at the bond market close while OAS uses ICE end-of-day marks (small timing mismatch).

    Raises:
        RuntimeError: if the download fails and no cache exists.
    """
    cache_path = _resolve_cache_dir(cache_dir) / f"{series_id}.csv"

    try:
        weekly = to_weekly(_download_fred(series_id, timeout).dropna()) / 100.0
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_csv(cache_path, header=True)
        return weekly
    except Exception as exc:  # noqa: BLE001 - any download failure triggers the fallback
        warnings.warn(
            f"FRED download of {series_id} failed ({exc!r}); using cache {cache_path}.",
            UserWarning,
            stacklevel=2,
        )

    if not cache_path.exists():
        raise RuntimeError(f"FRED download of {series_id} failed and no cache found at {cache_path}.")
    cached = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
    cached.index.name = "date"
    return cached.rename(series_id)


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def load_etf_characteristics(path: Path | str | None = None) -> pd.DataFrame:
    """Load the as-of dated ETF characteristics (duration, spread duration, convexity, hedge flag).

    The file is filled by hand from the official issuer factsheets. Expected columns:
    ticker, sleeve, duration, spread_duration, convexity, hedged, as_of.

    Column meaning:
      - ``hedged``: ``true`` means no residual currency risk for a USD investor, either
        because the fund holds USD assets or because it hedges its foreign-currency
        exposure back to USD (e.g. BNDX). ``false`` means unhedged FX exposure.
      - ``convexity``: optional. An empty cell becomes NaN with a ``UserWarning``; the
        convexity term is then ignored for that ETF (duration-only approximation).
        Every other column is mandatory.

    Returns a DataFrame indexed by ticker with float numeric columns, a boolean
    ``hedged`` column and a datetime ``as_of`` column.

    Assumptions: durations are effective/option-adjusted durations in years as
    published by the issuer, valid at ``as_of`` only.

    Limits: factsheet figures are point-in-time; they drift with markets and fund
    rebalancing, so they must be refreshed and their as-of date shown to the user.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if a column is missing, a mandatory value is empty, a ticker is
            duplicated or a value cannot be parsed.
    """
    path = Path(path) if path is not None else ETF_CHARACTERISTICS_PATH
    if not path.exists():
        raise FileNotFoundError(f"ETF characteristics file not found: {path}")

    df = pd.read_csv(path, dtype=str, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    missing_cols = [c for c in ETF_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path.name}: missing column(s): {', '.join(missing_cols)}")

    df = df[ETF_REQUIRED_COLUMNS].apply(lambda col: col.str.strip())
    empty = df.isna() | (df == "")

    for col in _ETF_OPTIONAL_COLUMNS:
        blank = df.loc[empty[col], "ticker"].tolist()
        if blank:
            warnings.warn(
                f"{path.name}: {col} is empty for {', '.join(map(str, blank))}; "
                f"set to NaN, the {col} term will be ignored for these ETFs.",
                UserWarning,
                stacklevel=2,
            )
        df.loc[empty[col], col] = np.nan
        empty[col] = False

    if empty.to_numpy().any():
        problems = [
            f"{df.at[i, 'ticker'] or f'row {i + 2}'}: {', '.join(empty.columns[empty.loc[i]])}"
            for i in df.index
            if empty.loc[i].any()
        ]
        raise ValueError(f"{path.name}: missing value(s) -> " + "; ".join(problems))

    duplicated = df["ticker"][df["ticker"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{path.name}: duplicated ticker(s): {', '.join(duplicated)}")

    for col in _ETF_NUMERIC_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = df.loc[numeric.isna() & df[col].notna(), "ticker"].tolist()
        if bad:
            raise ValueError(f"{path.name}: non-numeric {col} for: {', '.join(bad)}")
        df[col] = numeric.astype(float)

    try:
        df["hedged"] = df["hedged"].map(_parse_bool)
    except ValueError as exc:
        raise ValueError(f"{path.name}: invalid hedged value ({exc}); use true/false") from exc

    as_of = pd.to_datetime(df["as_of"], errors="coerce", format="%Y-%m-%d")
    bad = df.loc[as_of.isna(), "ticker"].tolist()
    if bad:
        raise ValueError(f"{path.name}: invalid as_of date (expected YYYY-MM-DD) for: {', '.join(bad)}")
    df["as_of"] = as_of

    return df.set_index("ticker")


def _coverage_row(name: str, source: str, series: pd.Series) -> dict[str, object]:
    valid = series.dropna()
    if valid.empty:
        return {"series": name, "source": source, "first_date": pd.NaT, "last_date": pd.NaT,
                "n_obs": 0, "n_missing": int(len(series))}
    window = series.loc[valid.index[0]: valid.index[-1]]
    return {
        "series": name,
        "source": source,
        "first_date": valid.index[0],
        "last_date": valid.index[-1],
        "n_obs": int(valid.size),
        "n_missing": int(window.isna().sum()),
    }


def data_coverage_report(
    prices: pd.DataFrame | None = None,
    fred: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Coverage of every weekly series: first date, last date, observations, missing values.

    If ``prices`` or ``fred`` are not given, they are loaded with ``get_prices()`` and
    ``get_fred_series()`` (which may fall back to the cache).

    Definitions: ``n_obs`` is the number of non-missing weekly values; ``n_missing``
    counts NaN weeks *between* the first and last valid dates (gaps), so weeks before
    an ETF's inception or after a series stops are not counted as missing.
    """
    if prices is None:
        prices = get_prices()
    if fred is None:
        fred = {sid: get_fred_series(sid) for sid in FRED_SERIES}

    rows = [_coverage_row(col, "yfinance", prices[col]) for col in prices.columns]
    rows += [_coverage_row(sid, "FRED", s) for sid, s in fred.items()]
    report = pd.DataFrame(rows).set_index("series")
    report["n_obs"] = report["n_obs"].astype(np.int64)
    report["n_missing"] = report["n_missing"].astype(np.int64)
    return report
