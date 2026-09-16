import pandas as pd

from backtest.mfe_profit_protection import MFEProfitProtectionBacktest


def _candles(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "atr"])


def test_activation_arms_break_even_for_next_bar_only():
    c = _candles([
        [0,100,100,100,100,10],
        [1,100,106,99,105,10],  # reaches +0.6R but BE applies only next bar
        [2,105,106,99,100,10],  # next bar touches entry -> protected stop
        [3,100,100,100,100,10],
    ])
    s = pd.DataFrame({"timestamp":[0], "decision":["LONG"]})
    t = MFEProfitProtectionBacktest(fee_bps_per_side=0, slippage_bps_per_side=0).run(c,s,rr=2)
    assert t.iloc[0]["outcome"] == "PROTECTED_STOP"
    assert t.iloc[0]["exit"] == 100


def test_tighten_after_one_r_locks_half_r_next_bar():
    c = _candles([
        [0,100,100,100,100,10],
        [1,100,111,100,110,10],  # +1.1R, tighten to +0.5R next bar
        [2,110,111,104,105,10],
        [3,105,105,105,105,10],
    ])
    s = pd.DataFrame({"timestamp":[0], "decision":["LONG"]})
    t = MFEProfitProtectionBacktest(fee_bps_per_side=0, slippage_bps_per_side=0).run(c,s,rr=2)
    assert t.iloc[0]["outcome"] == "PROTECTED_STOP"
    assert t.iloc[0]["exit"] == 105


def test_short_side_is_symmetric():
    c = _candles([
        [0,100,100,100,100,10],
        [1,100,100,89,90,10],
        [2,90,96,89,95,10],
        [3,95,95,95,95,10],
    ])
    s = pd.DataFrame({"timestamp":[0], "decision":["SHORT"]})
    t = MFEProfitProtectionBacktest(fee_bps_per_side=0, slippage_bps_per_side=0).run(c,s,rr=2)
    assert t.iloc[0]["outcome"] == "PROTECTED_STOP"
    assert t.iloc[0]["exit"] == 95
