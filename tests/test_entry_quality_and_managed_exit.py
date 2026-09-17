from __future__ import annotations

import pandas as pd

from backtest.entry_quality import apply_entry_quality
from backtest.managed_exit_backtest import ManagedExitBacktest


def test_entry_quality_requires_two_of_three_and_anti_chase():
    frame = pd.DataFrame(
        {
            "decision": ["LONG", "LONG", "SHORT"],
            "keltner_long_confirm": [True, True, False],
            "keltner_short_confirm": [False, False, True],
            "plus_di": [30.0, 30.0, 10.0],
            "minus_di": [10.0, 10.0, 30.0],
            "adx_rising": [True, False, True],
            "vwap_long_confirm": [False, True, False],
            "vwap_short_confirm": [False, False, True],
            "keltner_position": [0.5, 1.2, -0.7],
        }
    )
    out = apply_entry_quality(frame)
    assert out["decision"].tolist() == ["LONG", "NO_TRADE", "SHORT"]
    assert out["entry_quality_score"].tolist() == [2, 2, 3]


def test_break_even_only_activates_from_next_candle():
    candles = pd.DataFrame(
        [
            {"timestamp": "2026-09-01T00:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100, "atr": 10, "keltner_middle": 99},
            {"timestamp": "2026-09-01T00:15:00Z", "open": 100, "high": 112, "low": 99, "close": 111, "atr": 10, "keltner_middle": 102},
            {"timestamp": "2026-09-01T00:30:00Z", "open": 111, "high": 112, "low": 99, "close": 101, "atr": 10, "keltner_middle": 103},
        ]
    )
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    signals = pd.DataFrame({"timestamp": [candles.at[0, "timestamp"]], "decision": ["LONG"]})
    engine = ManagedExitBacktest(fee_bps_per_side=0, slippage_bps_per_side=0, structure_lookback=2)
    trades = engine.run(candles, signals, rr=3.0)
    assert len(trades) == 1
    assert trades.iloc[0]["outcome"] == "BREAKEVEN"
    assert trades.iloc[0]["exit_time"] == candles.at[2, "timestamp"]


def test_future_change_does_not_change_prior_entry_quality():
    frame = pd.DataFrame(
        {
            "decision": ["LONG", "LONG"],
            "keltner_long_confirm": [True, True],
            "keltner_short_confirm": [False, False],
            "plus_di": [30.0, 1.0],
            "minus_di": [10.0, 50.0],
            "adx_rising": [True, False],
            "vwap_long_confirm": [True, False],
            "vwap_short_confirm": [False, True],
            "keltner_position": [0.5, 2.0],
        }
    )
    a = apply_entry_quality(frame)
    altered = frame.copy()
    altered.loc[1, ["plus_di", "minus_di", "adx_rising", "vwap_long_confirm", "keltner_position"]] = [100, 0, True, True, 0.1]
    b = apply_entry_quality(altered)
    assert a.loc[0, "decision"] == b.loc[0, "decision"] == "LONG"
