from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig
from live import forward_engine as fe
from strategy.market_regime_gate import GateState, RegimeSnapshot, decide_regime


class ForwardEngineInvariantTests(unittest.TestCase):
    def test_official_protocol_integrity_and_config(self) -> None:
        cfg = IchimokuKeltner1HConfig()
        fe.assert_frozen_protocol(cfg)
        self.assertEqual(fe.protocol_digest(), fe.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(fe.INITIAL_EQUITY, 2000.0)
        self.assertEqual(fe.RISK_FRACTION, 0.10)
        self.assertEqual(fe.MAX_LEVERAGE, 15.0)

    def test_only_official_regimes_are_authorized(self) -> None:
        long_ok = decide_regime(RegimeSnapshot("LONG", "BULLISH", 0.01, 0.012, 0.018, 0.10))
        short_ok = decide_regime(RegimeSnapshot("SHORT", "BEARISH", 0.015, 0.010, 0.020, 0.20))
        long_bad = decide_regime(RegimeSnapshot("LONG", "BULLISH", 0.025, 0.012, 0.018, 0.10))
        self.assertIs(long_ok.state, GateState.ON)
        self.assertEqual(long_ok.regime_key, "LONG|LOW|CHOP")
        self.assertIs(short_ok.state, GateState.ON)
        self.assertEqual(short_ok.regime_key, "SHORT|NORMAL|MIXED")
        self.assertIs(long_bad.state, GateState.OFF)

    def test_dynamic_sizing_targets_ten_pct_and_caps_15x(self) -> None:
        distance, leverage, notional, risk = fe.size_position(2000.0, 2000.0, 1960.0)
        self.assertAlmostEqual(distance, 0.02)
        self.assertAlmostEqual(leverage, 5.0)
        self.assertAlmostEqual(notional, 10000.0)
        self.assertAlmostEqual(risk, 200.0)
        _, leverage_tight, _, _ = fe.size_position(2000.0, 2000.0, 1995.0)
        self.assertEqual(leverage_tight, 15.0)

    def test_entry_is_exact_next_candle_and_stores_sizing(self) -> None:
        state = fe.State(
            initialized=True,
            pending_side="LONG",
            pending_signal_time="2026-09-18T20:00:00+00:00",
            pending_signal_open_ms=1_700_000_000_000,
            pending_atr=20.0,
            pending_regime_key="LONG|LOW|CHOP",
        )
        row = pd.Series({
            "open_time_ms": fe.next_open_ms(state.pending_signal_open_ms),
            "timestamp": pd.Timestamp("2026-09-18T21:00:00"),
            "open": 2000.0,
        })
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(fe, "STATE_PATH", Path(tmp) / "state.json"):
                p = fe.open_position(state, row)
        self.assertEqual(p["stop_price"], 1960.0)
        self.assertAlmostEqual(p["leverage"], 5.0)
        self.assertAlmostEqual(p["notional_brl"], 10000.0)
        self.assertEqual(p["regime_key"], "LONG|LOW|CHOP")

    def test_same_bar_stop_wins_and_gap_uses_open(self) -> None:
        position = {"side": "LONG", "stop_price": 1950.0}
        row = pd.Series({"open": 1900.0, "high": 2100.0, "low": 1880.0})
        self.assertEqual(fe.evaluate_exit(position, row, 2050.0), (1900.0, "STOP 2ATR"))

    def test_close_uses_stored_dynamic_notional(self) -> None:
        cfg = IchimokuKeltner1HConfig()
        state = fe.State(
            initialized=True,
            equity_brl=2000.0,
            peak_equity_brl=2000.0,
            position={
                "side": "LONG",
                "regime_key": "LONG|LOW|CHOP",
                "signal_time": "2026-09-18T20:00:00+00:00",
                "entry_time": "2026-09-18T21:00:00+00:00",
                "entry_open_ms": 1,
                "entry_price": 2000.0,
                "entry_atr": 20.0,
                "stop_price": 1960.0,
                "stop_distance_pct": 2.0,
                "leverage": 5.0,
                "notional_brl": 10000.0,
                "risk_budget_brl": 200.0,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(fe, "STATE_PATH", root / "state.json"), patch.object(fe, "LEDGER_PATH", root / "trades.csv"):
                fe.close_position(state, pd.Timestamp("2026-09-18T22:00:00Z"), 2020.0, "KELTNER SUPERIOR 3.0", cfg)
                rows = fe.ledger_records()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["net_return_pct"]), 0.9, places=12)
        self.assertAlmostEqual(float(rows[0]["pnl_brl"]), 90.0, places=12)
        self.assertAlmostEqual(state.equity_brl, 2090.0, places=12)

    def test_market_data_continuity_validation(self) -> None:
        now_ms = 10 * fe.INTERVAL_MS
        base = pd.DataFrame({
            "open_time_ms": [7 * fe.INTERVAL_MS, 8 * fe.INTERVAL_MS, 9 * fe.INTERVAL_MS],
            "close_time_ms": [8 * fe.INTERVAL_MS - 1, 9 * fe.INTERVAL_MS - 1, 10 * fe.INTERVAL_MS - 1],
        })
        fe.validate_market_data(base, now_ms, fe.INTERVAL_MS)
        gap = base.copy()
        gap.loc[2, "open_time_ms"] += fe.INTERVAL_MS
        with self.assertRaisesRegex(RuntimeError, "continuity"):
            fe.validate_market_data(gap, now_ms, fe.INTERVAL_MS)

    def test_health_identifies_official_strategy(self) -> None:
        state = fe.State(initialized=True)
        now = datetime(2026, 9, 18, 20, 7, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            health = Path(tmp) / "health.json"
            with patch.object(fe, "HEALTH_PATH", health):
                fe.write_health(state, "OK", now, 123)
                payload = json.loads(health.read_text(encoding="utf-8"))
        self.assertEqual(payload["strategy"], "official_regime_risk_v1")
        self.assertEqual(payload["protocol_sha256"], fe.EXPECTED_PROTOCOL_SHA256)


if __name__ == "__main__":
    unittest.main()
