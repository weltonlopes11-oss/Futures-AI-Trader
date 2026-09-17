import pandas as pd

from backtest.exit_variants_backtest import ExitVariantsBacktest


def _frame():
    ts = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100,100,100,100,100,100,100,100],
        "high": [101,101,101,101,101.6,101.2,100.7,100.4],
        "low": [99,99,99,99,99.8,99.7,99.5,99.4],
        "close": [100,100,100,100,101.2,100.8,99.6,99.5],
        "atr": [1]*8,
        "keltner_middle": [100]*8,
    })


def test_three_bar_can_exit_before_five_bar():
    c = _frame()
    sig = pd.DataFrame({"timestamp": c["timestamp"], "decision": ["NO_TRADE","NO_TRADE","NO_TRADE","LONG","NO_TRADE","NO_TRADE","NO_TRADE","NO_TRADE"]})
    e = ExitVariantsBacktest(fee_bps_per_side=0, slippage_bps_per_side=0)
    t3 = e.run(c, sig, rr=5, mode="three_bar")
    t5 = e.run(c, sig, rr=5, mode="five_bar")
    assert len(t3) == 1 and len(t5) == 1
    assert t3.iloc[0]["exit_time"] <= t5.iloc[0]["exit_time"]


def test_keltner_middle_is_close_based():
    c = _frame()
    sig = pd.DataFrame({"timestamp": c["timestamp"], "decision": ["NO_TRADE","NO_TRADE","NO_TRADE","LONG","NO_TRADE","NO_TRADE","NO_TRADE","NO_TRADE"]})
    e = ExitVariantsBacktest(fee_bps_per_side=0, slippage_bps_per_side=0)
    t = e.run(c, sig, rr=5, mode="keltner_middle")
    assert len(t) == 1
    assert t.iloc[0]["outcome"] == "KELTNER_MIDDLE_EXIT"


def test_adaptive_arms_after_half_r():
    c = _frame()
    sig = pd.DataFrame({"timestamp": c["timestamp"], "decision": ["NO_TRADE","NO_TRADE","NO_TRADE","LONG","NO_TRADE","NO_TRADE","NO_TRADE","NO_TRADE"]})
    e = ExitVariantsBacktest(fee_bps_per_side=0, slippage_bps_per_side=0)
    t = e.run(c, sig, rr=5, mode="adaptive_05r_three_bar")
    assert len(t) == 1
    assert bool(t.iloc[0]["adaptive_armed_05r"])
