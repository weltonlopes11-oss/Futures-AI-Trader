from __future__ import annotations

import pandas as pd
import pytest

from backtest.fixed_historical_loader import FixedHistoricalDataLoader


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00",
                periods=5,
                freq="1min",
            ),
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 11, 12, 13, 14],
        }
    )


def test_fixed_loader_reads_and_sorts_dataset(tmp_path):
    path = tmp_path / "baseline.csv"

    data = _dataset().iloc[::-1]
    data.to_csv(path, index=False)

    loader = FixedHistoricalDataLoader(path)
    result = loader.load()

    assert len(result) == 5
    assert result.iloc[0]["timestamp"] == pd.Timestamp(
        "2026-01-01 00:00:00"
    )
    assert result.iloc[-1]["timestamp"] == pd.Timestamp(
        "2026-01-01 00:04:00"
    )


def test_fixed_loader_respects_limit(tmp_path):
    path = tmp_path / "baseline.csv"
    _dataset().to_csv(path, index=False)

    loader = FixedHistoricalDataLoader(path)
    result = loader.load(limit=3)

    assert len(result) == 3
    assert result.iloc[0]["timestamp"] == pd.Timestamp(
        "2026-01-01 00:02:00"
    )


def test_fixed_loader_rejects_missing_file(tmp_path):
    loader = FixedHistoricalDataLoader(
        tmp_path / "missing.csv"
    )

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_fixed_loader_rejects_invalid_schema(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "close": [1.0],
        }
    ).to_csv(path, index=False)

    loader = FixedHistoricalDataLoader(path)

    with pytest.raises(ValueError):
        loader.load()
