from __future__ import annotations

import pandas as pd

from backtest.structural_exit_backtest import StructuralExitBacktest


def _candles(rows):
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def test_long_exits_on_adverse_prior_structure_break():
    candles = _candles(
        [
            {"timestamp": "2026-09-01T00:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100, "atr": 10},
            {"timestamp": "2026-09-01T00:15:00Z", "open": 100, "high": 102, "low": 99, "close": 101, "atr": 10},
            {"timestamp": "2026-09-01T00:30:00Z", "open": 101, "high": 103, "low": 100, "close": 102, "atr": 10},
            {"timestamp": "2026-09-01T00:45:00Z", "open": 102, "high": 103, "low": 100, "close": 101, "atr": 10},
            {"timestamp": "2026-09-01T01:00:00Z", "open": 101, "high": 102, "low": 99, "close": 100, "atr": 10},
            {"timestamp": "2026-09-01T01:15:00Z", "open": 100, "high": 101, "low": 98, "close": 99, "atr": 10},
            {"timestamp": "2026-09-01T01:30:00Z", "open": 99, "high": 100, "low": 97, "close": 98, "atr": 10},
            {"timestamp": "2026-09-01T01:45:00Z", "open": 98, "high": 99, "low": 96, "close": 97, "atr": 10},
        ]
    )
    signals = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-09-01T00:00:00Z")], "decision": ["LONG"]}
    )

    engine = StructuralExitBacktest(fee_bps_per_side=0, slippage_bps_per_side=0, structure_lookback=3)
    trades = engine.run(candles, signals, rr=3.0)

    assert len(trades) == 1
    assert trades.iloc[0]["outcome"] == "STRUCTURE_EXIT"
    assert trades.iloc[0]["exit_time"] == pd.Timestamp("2026-09-01T01:15:00Z")


def test_future_candle_change_does_not_move_earlier_structure_exit():
    base = _candles(
        [
            {"timestamp": "2026-09-01T00:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100, "atr": 10},
            {"timestamp": "2026-09-01T00:15:00Z", "open": 100, "high": 102, "low": 99, "close": 101, "atr": 10},
            {"timestamp": "2026-09-01T00:30:00Z", "open": 101, "high": 103, "low": 100, "close": 102, "atr": 10},
            {"timestamp": "2026-09-01T00:45:00Z", "open": 102, "high": 103, "low": 100, "close": 101, "atr": 10},
            {"timestamp": "2026-09-01T01:00:00Z", "open": 101, "high": 102, "low": 99, "close": 98, "atr": 10},
            {"timestamp": "2026-09-01T01:15:00Z", "open": 98, "high": 150, "low": 50, "close": 120, "atr": 10},
        ]
    )
    altered = base.copy()
    altered.loc[altered.index[-1], ["high", "low", "close"]] = [500, 1, 400]
    signals = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-09-01T00:00:00Z")], "decision": ["LONG"]}
    )

    engine = StructuralExitBacktest(fee_bps_per_side=0, slippage_bps_per_side=0, structure_lookback=3)
    a = engine.run(base, signals, rr=5.0)
    b = engine.run(altered, signals, rr=5.0)

    assert a.iloc[0]["outcome"] == "STRUCTURE_EXIT"
    assert b.iloc[0]["outcome"] == "STRUCTURE_EXIT"
    assert a.iloc[0]["exit_time"] == b.iloc[0]["exit_time"]
    assert a.iloc[0]["exit"] == b.iloc[0]["exit"]
