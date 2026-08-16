from __future__ import annotations

import pandas as pd

from backtest.open_interest_enricher import OpenInterestEnricher


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-08-16 10:00:00",
                    "2026-08-16 10:01:00",
                    "2026-08-16 10:04:00",
                    "2026-08-16 10:05:00",
                    "2026-08-16 10:06:00",
                ]
            ),
            "open": [1, 1, 1, 1, 1],
            "high": [1, 1, 1, 1, 1],
            "low": [1, 1, 1, 1, 1],
            "close": [1, 1, 1, 1, 1],
            "volume": [1, 1, 1, 1, 1],
        }
    )


def _oi() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-08-16 10:00:00",
                    "2026-08-16 10:05:00",
                ]
            ),
            "sumOpenInterest": [100.0, 110.0],
            "sumOpenInterestValue": [300000.0, 330000.0],
        }
    )


def test_open_interest_uses_last_known_observation_only():
    result = OpenInterestEnricher().enrich(_candles(), _oi())
    by_time = result.set_index("timestamp")

    assert by_time.loc[
        pd.Timestamp("2026-08-16 10:04:00"),
        "open_interest",
    ] == 100.0

    assert by_time.loc[
        pd.Timestamp("2026-08-16 10:05:00"),
        "open_interest",
    ] == 110.0


def test_open_interest_change_is_calculated_on_observations_not_candles():
    result = OpenInterestEnricher().enrich(_candles(), _oi())
    by_time = result.set_index("timestamp")

    assert pd.isna(
        by_time.loc[
            pd.Timestamp("2026-08-16 10:04:00"),
            "open_interest_change_pct",
        ]
    )

    assert by_time.loc[
        pd.Timestamp("2026-08-16 10:05:00"),
        "open_interest_change_pct",
    ] == 10.0


def test_future_open_interest_does_not_leak_backwards():
    candles = _candles().iloc[:3].copy()

    oi = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-08-16 10:05:00"]),
            "sumOpenInterest": [110.0],
            "sumOpenInterestValue": [330000.0],
        }
    )

    result = OpenInterestEnricher().enrich(candles, oi)

    assert result["open_interest"].isna().all()


def test_binance_records_are_normalized_to_dataframe():
    records = [
        {
            "symbol": "ETHUSDT",
            "sumOpenInterest": "123.45",
            "sumOpenInterestValue": "456789.01",
            "timestamp": 1786884000000,
        }
    ]

    result = OpenInterestEnricher.from_binance_records(records)

    assert len(result) == 1
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
