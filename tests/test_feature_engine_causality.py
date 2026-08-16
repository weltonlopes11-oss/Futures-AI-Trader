from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from market.feature_engine import FeatureEngine


def _frame(start: str, periods: int) -> pd.DataFrame:
    timestamps = pd.date_range(start=start, periods=periods, freq="1min")
    values = pd.Series(range(periods), dtype="float64") + 1000.0

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": values,
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values + 0.5,
            "volume": 1.0,
        }
    )


def test_1h_feature_only_appears_at_hour_close():
    engine = FeatureEngine()
    data = _frame("2026-01-01 10:00:00", 61)

    result = engine._merge_timeframe(data, "1h", "trend_1h")
    by_time = result.set_index("timestamp")

    assert pd.isna(by_time.loc[pd.Timestamp("2026-01-01 10:59:00"), "trend_1h"])
    assert by_time.loc[pd.Timestamp("2026-01-01 11:00:00"), "trend_1h"] == "RANGE"


def test_4h_feature_only_appears_at_block_close():
    engine = FeatureEngine()
    data = _frame("2026-01-01 00:00:00", 241)

    result = engine._merge_timeframe(data, "4h", "trend_4h")
    by_time = result.set_index("timestamp")

    assert pd.isna(by_time.loc[pd.Timestamp("2026-01-01 03:59:00"), "trend_4h"])
    assert by_time.loc[pd.Timestamp("2026-01-01 04:00:00"), "trend_4h"] == "RANGE"


def test_1d_feature_only_appears_after_day_close():
    engine = FeatureEngine()
    data = _frame("2026-01-01 00:00:00", 1441)

    result = engine._merge_timeframe(data, "1d", "trend_1d")
    by_time = result.set_index("timestamp")

    assert pd.isna(by_time.loc[pd.Timestamp("2026-01-01 23:59:00"), "trend_1d"])
    assert by_time.loc[pd.Timestamp("2026-01-02 00:00:00"), "trend_1d"] == "RANGE"


def test_closed_higher_timeframe_is_available_exactly_on_right_edge():
    engine = FeatureEngine()
    data = _frame("2026-01-01 08:00:00", 121)

    result = engine._merge_timeframe(data, "1h", "trend_1h")
    by_time = result.set_index("timestamp")

    assert pd.isna(by_time.loc[pd.Timestamp("2026-01-01 08:59:00"), "trend_1h"])
    assert by_time.loc[pd.Timestamp("2026-01-01 09:00:00"), "trend_1h"] == "RANGE"


def test_appending_future_candles_does_not_change_past_features():
    engine = FeatureEngine()

    original = _frame("2026-01-01 00:00:00", 1501)
    extended = _frame("2026-01-01 00:00:00", 1801)

    original_result = engine.enrich(original).set_index("timestamp")
    extended_result = engine.enrich(extended).set_index("timestamp")

    common_index = original_result.index

    for column in ("trend_1h", "trend_4h", "trend_1d"):
        pdt.assert_series_equal(
            original_result[column],
            extended_result.loc[common_index, column],
            check_names=True,
        )
