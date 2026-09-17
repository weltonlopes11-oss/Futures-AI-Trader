from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from backtest.liquidity_signal_engine import LiquiditySignalConfig, LiquiditySignalEngine


def _series(direction: int, n: int = 40, start: str = "2026-09-16T12:00:00Z") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="2s", tz="UTC")
    base = 0.45 * direction
    return pd.DataFrame(
        {
            "timestamp": ts,
            "spread_bps": [0.4] * n,
            "depth_imbalance": [base + direction * (i / n) * 0.10 for i in range(n)],
            "microprice_edge_bps": [0.08 * direction] * n,
        }
    )


def test_persistent_buy_pressure_yields_long_signal():
    engine = LiquiditySignalEngine()
    result = engine.score_at(_series(1))
    assert result["liquidity_score"] > 0
    assert result["liquidity_signal"] == "LONG"
    assert result["persistent_imbalance"] > 0


def test_persistent_sell_pressure_yields_short_signal():
    engine = LiquiditySignalEngine()
    result = engine.score_at(_series(-1))
    assert result["liquidity_score"] < 0
    assert result["liquidity_signal"] == "SHORT"
    assert result["persistent_imbalance"] < 0


def test_wide_spread_blocks_even_with_directional_pressure():
    frame = _series(1)
    frame.loc[frame.index[-1], "spread_bps"] = 6.0
    engine = LiquiditySignalEngine(LiquiditySignalConfig(max_spread_bps=5.0))
    result = engine.score_at(frame)
    assert result["liquidity_signal"] == "BLOCK"


def test_score_is_causal_when_future_snapshots_are_appended():
    engine = LiquiditySignalEngine()
    base = _series(1, n=40)
    at = base["timestamp"].iloc[-1]
    expected = engine.score_at(base, at=at)

    future = _series(-1, n=20, start=(at + pd.Timedelta(seconds=2)).isoformat())
    extended = pd.concat([base, future], ignore_index=True)
    actual = engine.score_at(extended, at=at)

    keys = (
        "liquidity_score",
        "persistent_imbalance",
        "microprice_component",
        "imbalance_acceleration",
        "spread_quality",
        "directional_stability",
        "liquidity_signal",
    )
    for key in keys:
        assert actual[key] == expected[key]


def test_enrich_never_uses_future_rows():
    engine = LiquiditySignalEngine()
    base = _series(1, n=40)
    enriched_base = engine.enrich(base)

    future = _series(-1, n=20, start=(base["timestamp"].iloc[-1] + pd.Timedelta(seconds=2)).isoformat())
    enriched_extended = engine.enrich(pd.concat([base, future], ignore_index=True))

    overlap = enriched_extended[enriched_extended["timestamp"].isin(enriched_base["timestamp"])]
    pdt.assert_series_equal(
        enriched_base.set_index("timestamp")["liquidity_score"],
        overlap.set_index("timestamp")["liquidity_score"],
        check_names=True,
    )
