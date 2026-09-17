import pandas as pd

from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators, run_backtest


def candles(n=120, start=100.0):
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    close = pd.Series([start + i * 0.1 for i in range(n)], dtype=float)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
    })


def test_leading_spans_are_forward_displaced_and_causal():
    df = candles()
    out = enrich_indicators(df)
    raw_a = (out["tenkan"] + out["kijun"]) / 2.0
    expected = raw_a.shift(26)
    pd.testing.assert_series_equal(out["leading_span_a"], expected, check_names=False)


def test_executable_keltner_is_previous_completed_bar():
    out = enrich_indicators(candles())
    pd.testing.assert_series_equal(out["executable_keltner_upper"], out["keltner_upper"].shift(1), check_names=False)
    pd.testing.assert_series_equal(out["executable_keltner_lower"], out["keltner_lower"].shift(1), check_names=False)


def test_no_stop_loss_or_opposite_signal_forces_exit():
    # This test asserts the strategy contract: exits are Keltner touches or the
    # evaluation-boundary mark-to-market, never a protective stop/reversal.
    df = candles(140)
    trades = run_backtest(df, IchimokuKeltner1HConfig())
    if not trades.empty:
        assert set(trades["exit_reason"]).issubset({"KELTNER_UPPER_TOUCH", "KELTNER_LOWER_TOUCH", "EVAL_END_MTM"})
