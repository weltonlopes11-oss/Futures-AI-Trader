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
        self.assertEqual(fe.evaluate_exit(long_position, long_row, 2100.0), (1900.0, "STOP 2ATR"))
        short_position = {"side": "SHORT", "stop_price": 2050.0}
        short_row = pd.Series({"open": 2100.0, "high": 2120.0, "low": 2080.0})
        self.assertEqual(fe.evaluate_exit(short_position, short_row, 1900.0), (2100.0, "STOP 2ATR"))

    def test_10x_equity_accounting_uses_transactional_ledger_outbox(self) -> None:
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
            root = Path(tmp)
            with patch.object(fe, "STATE_PATH", root / "state.json"), patch.object(fe, "LEDGER_PATH", root / "trades.csv"):
                record = fe.close_position(
                    state,
                    pd.Timestamp("2026-09-17T22:00:00Z"),
                    2020.0,
                    "KELTNER SUPERIOR 3.0",
                    cfg,
                    notify=False,
                )
                fe.flush_ledger(state)
                rows = fe.ledger_records()
        self.assertEqual(len(rows), 1)
        self.assertEqual(state.ledger_outbox, [])
        self.assertAlmostEqual(record["net_return_pct"], 0.9, places=12)
        self.assertAlmostEqual(record["pnl_brl_10x"], 45.0, places=12)
        self.assertAlmostEqual(state.equity_brl, 545.0, places=12)
        self.assertEqual(state.trade_count, 1)

    def test_ledger_retry_is_idempotent_after_ambiguous_crash(self) -> None:
        record = {field: "" for field in fe.LEDGER_FIELDS}
        record.update({"trade_id": "LONG-1", "equity_after_brl": 500.0})
        state = fe.State(initialized=True, trade_count=1, ledger_outbox=[record.copy()])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(fe, "STATE_PATH", root / "state.json"), patch.object(fe, "LEDGER_PATH", root / "trades.csv"):
                fe.append_trade(record)
                fe.flush_ledger(state)
                fe.flush_ledger(state)
                rows = fe.ledger_records()
        self.assertEqual(len(rows), 1)
        self.assertEqual(state.ledger_outbox, [])

    def test_notification_outbox_deduplicates_normal_retries(self) -> None:
        state = fe.State(initialized=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(fe, "STATE_PATH", root / "state.json"), patch.object(fe, "telegram") as send:
                fe.queue_notification(state, "event-1", "hello")
                fe.queue_notification(state, "event-1", "hello")
                self.assertTrue(fe.flush_notifications(state))
                self.assertTrue(fe.flush_notifications(state))
        send.assert_called_once_with("hello")
        self.assertEqual(state.notification_outbox, {})
        self.assertIn("event-1", state.sent_notification_ids)

    def test_notification_failure_stays_in_outbox_without_blocking_state(self) -> None:
        state = fe.State(initialized=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(fe, "STATE_PATH", root / "state.json"), patch.object(fe, "telegram", side_effect=OSError("down")):
                fe.queue_notification(state, "event-1", "hello")
                self.assertFalse(fe.flush_notifications(state))
        self.assertEqual(state.notification_outbox, {"event-1": "hello"})

    def test_kill_switch_cancels_pending_and_is_auditable(self) -> None:
        state = fe.State(
            initialized=True,
            pending_side="LONG",
            pending_signal_time="2026-09-17T20:00:00+00:00",
            pending_signal_open_ms=1_700_000_000_000,
            pending_atr=20.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(fe, "STATE_PATH", Path(tmp) / "state.json"):
                allow = fe.apply_control(state, {"kill_switch": True, "reason": "operator test"})
        self.assertFalse(allow)
        self.assertTrue(state.kill_switch_active)
        self.assertIsNone(state.pending_side)
        self.assertIn("ops-kill-switch-active", state.notification_outbox)

    def test_reconciliation_detects_state_ledger_divergence(self) -> None:
        state = fe.State(initialized=True, trade_count=1)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(fe, "LEDGER_PATH", Path(tmp) / "trades.csv"):
                with self.assertRaisesRegex(RuntimeError, "state trades=1"):
                    fe.reconcile_state_and_ledger(state)

    def test_market_data_health_detects_gap_and_staleness(self) -> None:
        now_ms = 10 * fe.INTERVAL_MS
        base = pd.DataFrame(
            {
                "open_time_ms": [7 * fe.INTERVAL_MS, 8 * fe.INTERVAL_MS, 9 * fe.INTERVAL_MS],
                "close_time_ms": [8 * fe.INTERVAL_MS - 1, 9 * fe.INTERVAL_MS - 1, 10 * fe.INTERVAL_MS - 1],
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
            }
        )
        fe.validate_market_data(base, now_ms)
        gap = base.copy()
        gap.loc[2, "open_time_ms"] += fe.INTERVAL_MS
        with self.assertRaisesRegex(RuntimeError, "continuity"):
            fe.validate_market_data(gap, now_ms)
        stale = base.copy()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            fe.validate_market_data(stale, now_ms + 3 * fe.INTERVAL_MS)

    def test_health_file_contains_operational_snapshot(self) -> None:
        state = fe.State(initialized=True, last_success_utc="2026-09-17T22:00:00+00:00")
        now = datetime(2026, 9, 17, 22, 7, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            health = Path(tmp) / "health.json"
            with patch.object(fe, "HEALTH_PATH", health):
                fe.write_health(state, "OK", now, 123)
                payload = json.loads(health.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["latest_market_open_ms"], 123)
        self.assertEqual(payload["protocol_sha256"], fe.EXPECTED_PROTOCOL_SHA256)

    def test_old_state_schema_loads_with_reliability_defaults(self) -> None:
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
        self.assertEqual(state.ledger_outbox, [])
        self.assertEqual(state.notification_outbox, {})
        self.assertEqual(state.ops_status, "OK")
        self.assertIn("kill_switch_active", reloaded)


if __name__ == "__main__":
    unittest.main()
