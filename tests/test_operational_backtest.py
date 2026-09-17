import pandas as pd
from backtest.operational_backtest import OperationalBacktest


def test_enters_next_candle_and_hits_target():
    candles = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="15min"),
        "open": [100, 100, 101, 102],
        "high": [101, 101, 103, 103],
        "low": [99, 99.5, 100, 101],
        "close": [100, 100.5, 102, 102],
        "atr": [1, 1, 1, 1],
    })
    signals = pd.DataFrame({"timestamp": candles["timestamp"], "decision": ["LONG", "NO_TRADE", "NO_TRADE", "NO_TRADE"]})
    engine = OperationalBacktest(fee_bps_per_side=0, slippage_bps_per_side=0)
    trades = engine.run(candles, signals, rr=2.0)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] == candles.iloc[1]["timestamp"]
    assert trades.iloc[0]["entry"] == 100
    assert trades.iloc[0]["outcome"] == "TARGET"
    assert trades.iloc[0]["exit"] == 102


def test_same_candle_stop_and_target_is_conservative_stop():
    candles = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="15min"),
        "open": [100, 100, 100], "high": [101, 103, 101], "low": [99, 98, 99], "close": [100, 100, 100], "atr": [1, 1, 1],
    })
    signals = pd.DataFrame({"timestamp": candles["timestamp"], "decision": ["LONG", "NO_TRADE", "NO_TRADE"]})
    trades = OperationalBacktest(0, 0).run(candles, signals, rr=2.0)
    assert trades.iloc[0]["outcome"] == "STOP_BOTH_TOUCHED"
    assert trades.iloc[0]["exit"] == 99


def test_round_trip_cost_is_deducted():
    engine = OperationalBacktest(fee_bps_per_side=4, slippage_bps_per_side=1)
    assert engine.round_trip_cost_pct == 0.10
