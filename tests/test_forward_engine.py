from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig
from live import forward_engine as fe


class ForwardEngineInvariantTests(unittest.TestCase):
    def test_frozen_protocol_integrity_and_config(self) -> None:
        cfg = IchimokuKeltner1HConfig()
        fe.assert_frozen_protocol(cfg)
        self.assertEqual(fe.protocol_digest(), fe.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual((cfg.tenkan, cfg.kijun, cfg.senkou_b, cfg.displacement), (9, 26, 52, 26))
        self.assertEqual((cfg.keltner_ema, cfg.keltner_atr, cfg.keltner_multiplier), (20, 20, 3.0))
        self.assertEqual((cfg.fee_bps_per_side, cfg.slippage_bps_per_side), (4.0, 1.0))
        self.assertEqual(fe.LEVERAGE, 10.0)
        self.assertEqual(fe.INITIAL_EQUITY, 500.0)

    def test_entry_must_be_exact_next_candle_open(self) -> None:
        state = fe.State(initialized=True)
        signal_open = 1_700_000_000_000
        entry_open = fe.next_open_ms(signal_open)
        p = fe.open_position(
            state,
            "LONG",
            "2026-09-17T20:00:00+00:00",
            signal_open,
            pd.Timestamp("2026-09-17T21:00:00Z"),
            entry_open,
            2000.0,
            25.0,
        )
        self.assertEqual(p["entry_open_ms"], entry_open)
        self.assertEqual(p["stop_price"], 1950.0)
        with self.assertRaises(RuntimeError):
            fe.open_position(
                fe.State(initialized=True),
                "LONG",
                "2026-09-17T20:00:00+00:00",
                signal_open,
                pd.Timestamp("2026-09-17T22:00:00Z"),
                entry_open + fe.INTERVAL_MS,
                2000.0,
                25.0,
            )

    def test_same_bar_stop_wins_over_target(self) -> None:
        position = {"side": "LONG", "stop_price": 1950.0}
        row = pd.Series({"open": 2000.0, "high": 2100.0, "low": 1900.0})
        price, reason = fe.evaluate_exit(position, row, 2050.0)
        self.assertEqual(price, 1950.0)
        self.assertEqual(reason, "STOP 2ATR")

    def test_gap_beyond_stop_fills_at_open(self) -> None:
        long_position = {"side": "LONG", "stop_price": 1950.0}
        long_row = pd.Series({"open": 1900.0, "high": 1920.0, "low": 1880.0})
        price, reason = fe.evaluate_exit(long_position, long_row, 2100.0)
        self.assertEqual((price, reason), (1900.0, "STOP 2ATR"))

        short_position = {"side": "SHORT", "stop_price": 2050.0}
        short_row = pd.Series({"open": 2100.0, "high": 2120.0, "low": 2080.0})
        price, reason = fe.evaluate_exit(short_position, short_row, 1900.0)
        self.assertEqual((price, reason), (2100.0, "STOP 2ATR"))

    def test_10x_equity_accounting_and_ledger_are_idempotent(self) -> None:
        cfg = IchimokuKeltner1HConfig()
        state = fe.State(initialized=True)
        signal_open = 1_700_000_000_000
        fe.open_position(
            state,
            "LONG",
            "2026-09-17T20:00:00+00:00",
            signal_open,
            pd.Timestamp("2026-09-17T21:00:00Z"),
            fe.next_open_ms(signal_open),
            2000.0,
            20.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "trades.csv"
            with patch.object(fe, "LEDGER_PATH", ledger):
                record = fe.close_position(
                    state,
                    pd.Timestamp("2026-09-17T22:00:00Z"),
                    2020.0,
                    "KELTNER SUPERIOR 3.0",
                    cfg,
                    notify=False,
                )
                fe.append_trade(record)
                rows = ledger.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(record["net_return_pct"], 0.9, places=12)
        self.assertAlmostEqual(record["pnl_brl_10x"], 45.0, places=12)
        self.assertAlmostEqual(state.equity_brl, 545.0, places=12)
        self.assertEqual(state.trade_count, 1)

    def test_old_state_schema_loads_with_new_defaults(self) -> None:
        old = {
            "equity_brl": 500.0,
            "peak_equity_brl": 500.0,
            "max_drawdown_pct": 0.0,
            "last_processed_open_ms": 1789675200000,
            "pending_side": None,
            "pending_signal_time": None,
            "position": None,
            "trade_count": 0,
            "initialized": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            state_path.write_text(json.dumps(old), encoding="utf-8")
            with patch.object(fe, "STATE_PATH", state_path):
                state = fe.load_state()
                fe.save_state(state)
                reloaded = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertTrue(state.initialized)
        self.assertFalse(state.finalized)
        self.assertIsNone(state.pending_signal_open_ms)
        self.assertIn("finalized", reloaded)
        self.assertIn("pending_atr", reloaded)


if __name__ == "__main__":
    unittest.main()
