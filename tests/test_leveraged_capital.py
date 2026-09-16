from __future__ import annotations

import pandas as pd

from backtest.leveraged_capital import LeveragedCapitalConfig, LeveragedCapitalSimulator


def test_compounds_net_returns_with_leverage():
    candles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-09-01 00:15Z", "2026-09-01 00:30Z", "2026-09-01 00:45Z"]),
            "high": [101.0, 102.0, 101.0],
            "low": [99.0, 100.0, 98.0],
        }
    )
    trades = pd.DataFrame(
        [
            {
                "entry_time": pd.Timestamp("2026-09-01 00:15Z"),
                "exit_time": pd.Timestamp("2026-09-01 00:30Z"),
                "side": "LONG",
                "entry": 100.0,
                "net_return_pct": 1.0,
                "outcome": "TARGET",
            },
            {
                "entry_time": pd.Timestamp("2026-09-01 00:45Z"),
                "exit_time": pd.Timestamp("2026-09-01 00:45Z"),
                "side": "LONG",
                "entry": 100.0,
                "net_return_pct": -0.5,
                "outcome": "STOP",
            },
        ]
    )
    sim = LeveragedCapitalSimulator(LeveragedCapitalConfig(initial_capital=500, leverage=10, maintenance_margin_rate=0.005))
    out = sim.simulate(candles, trades)
    assert round(out.iloc[0]["capital_after"], 2) == 550.00
    assert round(out.iloc[1]["capital_after"], 2) == 522.50


def test_detects_liquidation_threshold_cross():
    candles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-09-01 00:15Z"]),
            "high": [100.0],
            "low": [90.0],
        }
    )
    trades = pd.DataFrame(
        [
            {
                "entry_time": pd.Timestamp("2026-09-01 00:15Z"),
                "exit_time": pd.Timestamp("2026-09-01 00:15Z"),
                "side": "LONG",
                "entry": 100.0,
                "net_return_pct": -1.0,
                "outcome": "STOP",
            }
        ]
    )
    sim = LeveragedCapitalSimulator(LeveragedCapitalConfig(initial_capital=500, leverage=10, maintenance_margin_rate=0.005))
    out = sim.simulate(candles, trades)
    assert bool(out.iloc[0]["approx_liquidation_hit"])
    assert out.iloc[0]["capital_after"] == 0.0


def test_short_adverse_excursion_is_high_above_entry():
    candles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-09-01 00:15Z"]),
            "high": [104.0],
            "low": [97.0],
        }
    )
    trades = pd.DataFrame(
        [
            {
                "entry_time": pd.Timestamp("2026-09-01 00:15Z"),
                "exit_time": pd.Timestamp("2026-09-01 00:15Z"),
                "side": "SHORT",
                "entry": 100.0,
                "net_return_pct": 0.5,
                "outcome": "TARGET",
            }
        ]
    )
    sim = LeveragedCapitalSimulator()
    out = sim.simulate(candles, trades)
    assert round(out.iloc[0]["max_adverse_price_pct"], 6) == 4.0
    assert not bool(out.iloc[0]["approx_liquidation_hit"])
