import pandas as pd

from backtest.trade_quality_analytics import TradeQualityAnalytics


def test_long_mfe_mae_are_normalized_by_initial_risk():
    candles = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-01T00:15Z","2026-09-01T00:30Z","2026-09-01T00:45Z"]),
        "high": [101.0, 103.0, 102.0],
        "low": [99.5, 100.0, 98.0],
        "close": [100.5, 102.0, 99.0],
    })
    context = pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-01T00:00Z"]), "adx": [20.0]})
    trades = pd.DataFrame([{
        "signal_time": pd.Timestamp("2026-09-01T00:00Z"),
        "entry_time": pd.Timestamp("2026-09-01T00:15Z"),
        "exit_time": pd.Timestamp("2026-09-01T00:45Z"),
        "side": "LONG", "entry": 100.0, "stop": 98.0, "net_return_pct": 1.0,
    }])
    out = TradeQualityAnalytics().enrich_trades(candles, context, trades)
    assert out.iloc[0]["mfe_r"] == 1.5
    assert out.iloc[0]["mae_r"] == 1.0
    assert out.iloc[0]["bars_to_mfe"] == 1
    assert out.iloc[0]["bars_to_mae"] == 2


def test_short_mfe_mae_direction_is_correct():
    candles = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-01T00:15Z","2026-09-01T00:30Z"]),
        "high": [101.0, 102.0], "low": [99.0, 96.0], "close": [100.0, 97.0],
    })
    context = pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-01T00:00Z"])})
    trades = pd.DataFrame([{
        "signal_time": pd.Timestamp("2026-09-01T00:00Z"),
        "entry_time": pd.Timestamp("2026-09-01T00:15Z"),
        "exit_time": pd.Timestamp("2026-09-01T00:30Z"),
        "side": "SHORT", "entry": 100.0, "stop": 102.0, "net_return_pct": 2.0,
    }])
    out = TradeQualityAnalytics().enrich_trades(candles, context, trades)
    assert out.iloc[0]["mfe_r"] == 2.0
    assert out.iloc[0]["mae_r"] == 1.0


def test_entry_features_use_signal_time_not_future_bars():
    candles = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-01T00:15Z"]), "high": [101.0], "low": [99.0], "close": [100.0]
    })
    context = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-01T00:00Z","2026-09-01T00:15Z"]),
        "adx": [10.0, 99.0], "structure_regime": ["BULLISH", "BEARISH"],
    })
    trades = pd.DataFrame([{
        "signal_time": pd.Timestamp("2026-09-01T00:00Z"),
        "entry_time": pd.Timestamp("2026-09-01T00:15Z"),
        "exit_time": pd.Timestamp("2026-09-01T00:15Z"),
        "side": "LONG", "entry": 100.0, "stop": 99.0, "net_return_pct": 0.1,
    }])
    out = TradeQualityAnalytics().enrich_trades(candles, context, trades)
    assert out.iloc[0]["entry_adx"] == 10.0
    assert out.iloc[0]["entry_structure_regime"] == "BULLISH"
