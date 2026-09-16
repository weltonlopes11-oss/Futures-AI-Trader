from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from backtest.bybit_derivatives_history import BybitDerivativesHistoryLoader


def test_normalize_oi_sorts_and_computes_change():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    raw = pd.DataFrame(
        [
            {"timestamp": 1788221700000, "openInterest": "110"},
            {"timestamp": 1788220800000, "openInterest": "100"},
        ]
    )
    out = BybitDerivativesHistoryLoader._normalize_oi(raw, start, end)
    assert out.loc[0, "bybit_open_interest"] == 100.0
    assert out.loc[1, "bybit_open_interest"] == 110.0
    assert abs(out.loc[1, "bybit_open_interest_change_pct"] - 0.10) < 1e-12


def test_normalize_funding_preserves_source_timestamp():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    raw = pd.DataFrame(
        [
            {"fundingRateTimestamp": 1788220800000, "fundingRate": "0.0001"},
        ]
    )
    out = BybitDerivativesHistoryLoader._normalize_funding(raw, start, end)
    assert out.loc[0, "bybit_funding_rate"] == 0.0001
    assert out.loc[0, "bybit_funding_source_timestamp"] == pd.Timestamp(
        "2026-09-01T00:00:00Z"
    )


def test_oi_alignment_is_backward_only():
    candles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-09-01T00:14:00Z", "2026-09-01T00:16:00Z"], utc=True
            )
        }
    )
    oi = pd.DataFrame(
        {
            "bybit_oi_source_timestamp": pd.to_datetime(
                ["2026-09-01T00:15:00Z"], utc=True
            ),
            "bybit_open_interest": [100.0],
            "bybit_open_interest_change_pct": [0.01],
        }
    )
    out = BybitDerivativesHistoryLoader.align_oi_causally(candles, oi)
    assert pd.isna(out.loc[0, "bybit_open_interest"])
    assert out.loc[1, "bybit_open_interest"] == 100.0


def test_funding_alignment_is_backward_only():
    candles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-09-01T07:59:00Z", "2026-09-01T08:01:00Z"], utc=True
            )
        }
    )
    funding = pd.DataFrame(
        {
            "bybit_funding_source_timestamp": pd.to_datetime(
                ["2026-09-01T08:00:00Z"], utc=True
            ),
            "bybit_funding_rate": [0.0002],
            "bybit_funding_z": [1.25],
        }
    )
    out = BybitDerivativesHistoryLoader.align_funding_causally(candles, funding)
    assert pd.isna(out.loc[0, "bybit_funding_rate"])
    assert out.loc[1, "bybit_funding_rate"] == 0.0002
    assert out.loc[1, "bybit_funding_z"] == 1.25
