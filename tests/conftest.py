import pytest

from core import data


def _network_disabled(*args, **kwargs):
    raise RuntimeError("network access is disabled in tests")


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    """Block yfinance and requests, and point the cache to a temporary directory."""
    monkeypatch.setattr(data.yf, "download", _network_disabled)
    monkeypatch.setattr(data.requests, "get", _network_disabled)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
