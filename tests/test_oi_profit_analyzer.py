from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from backtest.oi_profit_analyzer import OIProfitAnalyzer


def _record(timestamp, oi, change, decision="LONG", regime="STRONG_BULL"):
    return SimpleNamespace(
        timestamp=pd.Timestamp(timestamp),
        close_price=100.0,
        market=SimpleNamespace(
            regime=regime,
            direction="LONG" if decision == "LONG" else "SHORT",
            metadata={
                "open_interest": oi,
                "open_interest_change_pct": change,
            },
        ),
        decision=SimpleNamespace(action=decision),
    )


def _candles(minutes=70):
    timestamps = pd.date_range("2026-08-16 10:00:00", periods=minutes, freq="min")
    close = [100.0 + index * 0.1 for index in range(minutes)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": [1.0] * minutes,
        }
    )


def test_analyzer_uses_only_real_oi_observations():
    records = [
        _record("2026-08-16 10:00:00", 100.0, None),
        _record("2026-08-16 10:01:00", 100.0, None),
        _record("2026-08-16 10:05:00", 110.0, 10.0),
        _record("2026-08-16 10:06:00", 110.0, 10.0),
        _record("2026-08-16 10:10:00", 99.0, -10.0),
    ]

    analyzer = OIProfitAnalyzer(records, _candles(), horizons=(5,))

    assert len(analyzer.base_observations) == 2
    assert len(analyzer.observations) == 2


def test_entry_uses_next_candle_open():
    records = [
        _record("2026-08-16 10:05:00", 110.0, 10.0),
        _record("2026-08-16 10:10:00", 99.0, -10.0),
    ]

    candles = _candles()
    analyzer = OIProfitAnalyzer(records, candles, horizons=(5,))
    item = analyzer.observations[0]

    expected = candles.loc[
        candles["timestamp"] == pd.Timestamp("2026-08-16 10:06:00"),
        "open",
    ].iloc[0]

    assert item.entry_timestamp == pd.Timestamp("2026-08-16 10:06:00")
    assert item.entry_price == pytest.approx(expected)


def test_horizons_calculate_future_return_mfe_and_mae():
    records = [
        _record("2026-08-16 10:05:00", 110.0, 10.0),
        _record("2026-08-16 10:10:00", 99.0, -10.0),
    ]

    analyzer = OIProfitAnalyzer(records, _candles(), horizons=(5,))
    item = analyzer.observations[0]

    assert item.horizon_minutes == 5
    assert item.future_return_pct > 0
    assert item.directional_return_pct > 0
    assert item.mfe_pct > 0
    assert item.mae_pct >= 0


def test_short_directional_return_inverts_price_return():
    records = [
        _record("2026-08-16 10:05:00", 110.0, 10.0, decision="SHORT"),
        _record("2026-08-16 10:10:00", 99.0, -10.0, decision="SHORT"),
    ]

    analyzer = OIProfitAnalyzer(records, _candles(), horizons=(5,))
    item = analyzer.observations[0]

    assert item.future_return_pct > 0
    assert item.directional_return_pct < 0


def test_round_trip_cost_is_deducted_from_trade_return():
    records = [
        _record("2026-08-16 10:05:00", 110.0, 10.0),
        _record("2026-08-16 10:10:00", 99.0, -10.0),
    ]

    analyzer = OIProfitAnalyzer(
        records,
        _candles(),
        horizons=(5,),
        fee_bps_per_side=5.0,
        slippage_bps_per_side=2.0,
    )
    item = analyzer.observations[0]

    assert analyzer.round_trip_cost_pct == pytest.approx(0.14)
    assert item.net_return_pct == pytest.approx(
        item.directional_return_pct - 0.14
    )


def test_bucket_thresholds_are_data_driven():
    records = []
    oi = 100.0
    changes = [-10, -5, -1, 0.5, 1, 5, 10]

    for index, change in enumerate(changes):
        oi *= 1 + change / 100.0
        records.append(
            _record(
                pd.Timestamp("2026-08-16 10:00:00") + pd.Timedelta(minutes=index * 5),
                oi,
                change,
            )
        )

    analyzer = OIProfitAnalyzer(records, _candles(), horizons=(5,))

    assert analyzer.thresholds["p10"] < analyzer.thresholds["p90"]
    assert analyzer.bucket_for(-10) == "STRONG_DROP"
    assert analyzer.bucket_for(10) == "STRONG_RISE"


def test_incomplete_future_window_is_discarded():
    records = [
        _record("2026-08-16 11:08:00", 110.0, 1.0),
        _record("2026-08-16 11:09:00", 111.0, 1.0),
    ]

    analyzer = OIProfitAnalyzer(records, _candles(minutes=70), horizons=(5,))

    assert analyzer.observations == []


def test_summary_contains_profit_quality_metrics():
    records = [
        _record("2026-08-16 10:05:00", 110.0, 10.0),
        _record("2026-08-16 10:10:00", 99.0, -10.0),
        _record("2026-08-16 10:15:00", 105.0, 6.0),
    ]

    analyzer = OIProfitAnalyzer(records, _candles(), horizons=(5,))
    summary = analyzer.summary_by_bucket()[5]

    assert summary
    first = next(iter(summary.values()))
    assert "avg_directional_return_pct" in first
    assert "avg_net_return_pct" in first
    assert "median_net_return_pct" in first
    assert "net_positive_rate_pct" in first
    assert "avg_mfe_pct" in first
    assert "avg_mae_pct" in first
    assert "mfe_mae_ratio" in first
