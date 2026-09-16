from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from backtest.binance_positioning_context import BinancePositioningContextLoader
from backtest.crypto_leverage_stress import CryptoLeverageStressConfig, CryptoLeverageStressScore
from backtest.cross_exchange_derivatives import CrossExchangeDerivatives
from backtest.liquidation_pressure import LiquidationPressure


def test_premium_uses_close_time_for_causality(monkeypatch):
    loader = BinancePositioningContextLoader()

    def fake_get(path, params):
        assert path == "/fapi/v1/premiumIndexKlines"
        return [[1000, "0.1", "0.2", "0.0", "0.15", "0", 1999, "0", 1, "0", "0", "0"]]

    # Force the deterministic unit test through the REST fallback. Production
    # historical research prefers Data Vision, which avoids runner geo-blocks.
    monkeypatch.setattr(loader, "_fetch_premium_data_vision_day", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(loader, "_get", fake_get)
    frame = loader.fetch_premium_index(
        "ETHUSDT",
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 2, tzinfo=timezone.utc),
    )
    assert frame.loc[0, "timestamp"].value // 1_000_000 == 1999
    assert frame.loc[0, "premium_open_time"].value // 1_000_000 == 1000
    assert frame.loc[0, "premium_close"] == 0.15


def test_liquidation_pressure_maps_sell_to_long_liquidation():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-09-16T00:00:00Z", "2026-09-16T00:00:30Z"], utc=True
            ),
            "side": ["SELL", "BUY"],
            "price": [2000.0, 2000.0],
            "quantity": [2.0, 1.0],
        }
    )
    out = LiquidationPressure().aggregate(events)
    assert out.loc[0, "long_liq"] == 4000.0
    assert out.loc[0, "short_liq"] == 2000.0
    assert out.loc[0, "liquidation_imbalance"] < 0


def test_leverage_stress_is_causal_and_detects_long_crowding():
    n = 120
    ts = pd.date_range("2026-09-01", periods=n, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "close": range(2000, 2000 + n),
            "open_interest_change_pct": [0.01] * 96 + [0.20] * 24,
            "funding_rate": [0.0001] * 96 + [0.0010] * 24,
            "premium_close": [0.0001] * 96 + [0.0015] * 24,
            "global_ls_ratio": [1.0] * 96 + [1.8] * 24,
            "top_account_ls_ratio": [1.0] * 96 + [1.9] * 24,
            "top_position_ls_ratio": [1.0] * 96 + [2.0] * 24,
            "cvd_delta": [1.0] * 96 + [10.0] * 24,
        }
    )
    engine = CryptoLeverageStressScore(
        CryptoLeverageStressConfig(min_periods=20, rolling_window=48)
    )
    out = engine.enrich(frame)
    assert out.loc[119, "crypto_leverage_stress_score"] > 0

    modified = frame.copy()
    modified.loc[119, "funding_rate"] = 99.0
    earlier_a = engine.enrich(frame).loc[:118, "crypto_leverage_stress_score"]
    earlier_b = engine.enrich(modified).loc[:118, "crypto_leverage_stress_score"]
    pd.testing.assert_series_equal(earlier_a, earlier_b)


def test_cross_exchange_aggregate_uses_only_normalized_oi():
    frame = pd.DataFrame(
        [
            {"oi_usd_estimate": 100.0, "funding_rate": 0.01},
            {"oi_usd_estimate": 300.0, "funding_rate": -0.01},
            {"oi_usd_estimate": None, "funding_rate": 9.0},
        ]
    )
    result = CrossExchangeDerivatives.aggregate(frame)
    assert result["global_oi_usd"] == 400.0
    assert abs(result["oi_weighted_funding"] - (-0.005)) < 1e-12
