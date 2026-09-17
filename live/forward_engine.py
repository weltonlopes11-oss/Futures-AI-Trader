from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators
from live.market_data import fetch_klines

STATE_PATH = Path(os.getenv("FORWARD_STATE_PATH", "live/state.json"))
LEDGER_PATH = Path(os.getenv("FORWARD_LEDGER_PATH", "live/trades.csv"))
HEALTH_PATH = Path(os.getenv("FORWARD_HEALTH_PATH", "live/health.json"))
CONTROL_PATH = Path(os.getenv("FORWARD_CONTROL_PATH", "live/control.json"))
PROTOCOL_PATH = Path(
    os.getenv(
        "FROZEN_PROTOCOL_PATH",
        "research_protocol/frozen_ichimoku_keltner_1h_10x_v1_2026-09-17.json",
    )
)
EXPECTED_PROTOCOL_SHA256 = "359b92e6024c0b3350775e53281c783cbcbaf68e030f3fc1139e7df580fce203"
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
INTERVAL_MS = 60 * 60 * 1000
LEVERAGE = 10.0
INITIAL_EQUITY = 500.0
START_UTC = pd.Timestamp(os.getenv("FORWARD_START_UTC", "2026-09-17T00:00:00Z"))
END_UTC = pd.Timestamp(os.getenv("FORWARD_END_UTC", "2026-10-17T23:59:59Z"))
HEARTBEAT_AFTER_UTC_HOUR = 12
MAX_SENT_NOTIFICATION_IDS = 500
LEDGER_FIELDS = (
    "trade_id",
    "side",
    "signal_time",
    "entry_time",
    "entry_price",
    "entry_atr",
    "stop_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "net_return_pct",
    "pnl_brl_10x",
    "equity_before_brl",
    "equity_after_brl",
)


@dataclass
class State:
    equity_brl: float = INITIAL_EQUITY
    peak_equity_brl: float = INITIAL_EQUITY
    max_drawdown_pct: float = 0.0
    last_processed_open_ms: int = 0
    pending_side: str | None = None
    pending_signal_time: str | None = None
    pending_signal_open_ms: int | None = None
    pending_atr: float | None = None
    position: dict | None = None
    trade_count: int = 0
    initialized: bool = False
    finalized: bool = False
    ledger_outbox: list[dict] = field(default_factory=list)
    notification_outbox: dict[str, str] = field(default_factory=dict)
    sent_notification_ids: list[str] = field(default_factory=list)
    ops_status: str = "OK"
    last_error_fingerprint: str | None = None
    last_success_utc: str | None = None
    last_heartbeat_date: str | None = None
    kill_switch_active: bool = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_state() -> State:
    if not STATE_PATH.exists():
        return State()
    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    allowed = State.__dataclass_fields__.keys()
    state = State(**{key: value for key, value in payload.items() if key in allowed})
    if not isinstance(state.ledger_outbox, list):
        raise RuntimeError("invalid state: ledger_outbox must be a list")
    if not isinstance(state.notification_outbox, dict):
        raise RuntimeError("invalid state: notification_outbox must be an object")
    if not isinstance(state.sent_notification_ids, list):
        raise RuntimeError("invalid state: sent_notification_ids must be a list")
    return state


def save_state(state: State) -> None:
    atomic_write_json(STATE_PATH, asdict(state))


def protocol_digest(path: Path = PROTOCOL_PATH) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def assert_frozen_protocol(cfg: IchimokuKeltner1HConfig, path: Path = PROTOCOL_PATH) -> None:
    digest = protocol_digest(path)
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"frozen protocol integrity failure: {digest}")
    if cfg != IchimokuKeltner1HConfig():
        raise RuntimeError("runtime strategy config differs from frozen protocol")


def telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body)
    with urllib.request.urlopen(req, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"Telegram HTTP {response.status}")


def queue_notification(state: State, event_id: str, text: str) -> None:
    if event_id in state.sent_notification_ids or event_id in state.notification_outbox:
        return
    state.notification_outbox[event_id] = text
    save_state(state)


def flush_notifications(state: State) -> bool:
    all_sent = True
    for event_id, text in list(state.notification_outbox.items()):
        try:
            telegram(text)
        except Exception:
            all_sent = False
            continue
        state.notification_outbox.pop(event_id, None)
        state.sent_notification_ids.append(event_id)
        state.sent_notification_ids = state.sent_notification_ids[-MAX_SENT_NOTIFICATION_IDS:]
        save_state(state)
    return all_sent


def load_control() -> dict:
    if not CONTROL_PATH.exists():
        return {"kill_switch": False, "reason": ""}
    payload = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload.get("kill_switch"), bool):
        raise RuntimeError("invalid control: kill_switch must be boolean")
    return {
        "kill_switch": payload["kill_switch"],
        "reason": str(payload.get("reason", "")).strip(),
    }


def apply_control(state: State, control: dict) -> bool:
    active = bool(control["kill_switch"])
    reason = control.get("reason") or "sem motivo informado"
    if active and not state.kill_switch_active:
        state.pending_side = None
        state.pending_signal_time = None
        state.pending_signal_open_ms = None
        state.pending_atr = None
        state.kill_switch_active = True
        queue_notification(
            state,
            "ops-kill-switch-active",
            f"KILL SWITCH ATIVO 🛑\nNovas entradas bloqueadas.\nPosições já abertas continuam sob stop/alvo congelados.\nMotivo: {reason}",
        )
    elif not active and state.kill_switch_active:
        state.kill_switch_active = False
        queue_notification(
            state,
            f"ops-kill-switch-released-{utc_now().date().isoformat()}",
            "KILL SWITCH LIBERADO ✅\nO forward voltou a aceitar novos sinais a partir deste momento.",
        )
    return not active


def net_return_pct(side: str, entry: float, exit_price: float, cfg: IchimokuKeltner1HConfig) -> float:
    gross = (exit_price / entry - 1.0) if side == "LONG" else (entry / exit_price - 1.0)
    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 10000.0
    return (gross - cost) * 100.0


def next_open_ms(signal_open_ms: int) -> int:
    return signal_open_ms + INTERVAL_MS


def open_position(
    state: State,
    side: str,
    signal_time: str,
    signal_open_ms: int,
    entry_time: pd.Timestamp,
    entry_open_ms: int,
    entry_price: float,
    atr: float,
) -> dict:
    if entry_open_ms != next_open_ms(signal_open_ms):
        raise RuntimeError("entry timing invariant failed: execution is not signal t -> open t+1")
    stop = entry_price - 2.0 * atr if side == "LONG" else entry_price + 2.0 * atr
    position = {
        "side": side,
        "entry_time": entry_time.isoformat(),
        "entry_open_ms": entry_open_ms,
        "entry_price": entry_price,
        "entry_atr": atr,
        "stop_price": stop,
        "signal_time": signal_time,
        "signal_open_ms": signal_open_ms,
        "entry_equity_brl": state.equity_brl,
    }
    state.position = position
    state.pending_side = None
    state.pending_signal_time = None
    state.pending_signal_open_ms = None
    state.pending_atr = None
    return position


def evaluate_exit(position: dict, row: pd.Series, target: float | None) -> tuple[float | None, str | None]:
    side = position["side"]
    stop = float(position["stop_price"])
    o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
    if side == "LONG":
        if l <= stop:
            return min(o, stop), "STOP 2ATR"
        if target is not None and pd.notna(target) and h >= float(target):
            return max(o, float(target)), "KELTNER SUPERIOR 3.0"
    else:
        if h >= stop:
            return max(o, stop), "STOP 2ATR"
        if target is not None and pd.notna(target) and l <= float(target):
            return min(o, float(target)), "KELTNER INFERIOR 3.0"
    return None, None


def ledger_records() -> list[dict]:
    if not LEDGER_PATH.exists() or LEDGER_PATH.stat().st_size == 0:
        return []
    with LEDGER_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_trade(record: dict) -> bool:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if any(row.get("trade_id") == record["trade_id"] for row in ledger_records()):
        return False
    exists = LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0
    with LEDGER_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in LEDGER_FIELDS})
    return True


def flush_ledger(state: State) -> None:
    for record in list(state.ledger_outbox):
        append_trade(record)
        state.ledger_outbox.remove(record)
        save_state(state)


def reconcile_state_and_ledger(state: State) -> None:
    rows = ledger_records()
    trade_ids = [row.get("trade_id") for row in rows]
    if len(trade_ids) != len(set(trade_ids)):
        raise RuntimeError("reconciliation failure: duplicate trade_id in ledger")
    if state.trade_count != len(rows):
        raise RuntimeError(f"reconciliation failure: state trades={state.trade_count}, ledger trades={len(rows)}")
    if rows:
        ledger_equity = float(rows[-1]["equity_after_brl"])
        if not math.isclose(ledger_equity, state.equity_brl, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError("reconciliation failure: ledger equity differs from state equity")
    if state.position is not None:
        current_id = f"{state.position['side']}-{int(state.position['entry_open_ms'])}"
        if current_id in trade_ids:
            raise RuntimeError("reconciliation failure: open position already exists as closed trade")
    pending_values = (
        state.pending_side,
        state.pending_signal_time,
        state.pending_signal_open_ms,
        state.pending_atr,
    )
    pending_count = sum(value is not None for value in pending_values)
    if pending_count not in (0, 4):
        raise RuntimeError("reconciliation failure: incomplete pending signal")
    if state.position is not None and pending_count:
        raise RuntimeError("reconciliation failure: position and pending signal coexist")
    if state.peak_equity_brl + 1e-8 < state.equity_brl:
        raise RuntimeError("reconciliation failure: peak equity below current equity")


def validate_market_data(raw: pd.DataFrame, now_ms: int) -> None:
    required = {"open_time_ms", "close_time_ms", "open", "high", "low", "close"}
    missing = required.difference(raw.columns)
    if missing:
        raise RuntimeError(f"market data missing columns: {sorted(missing)}")
    if raw.empty:
        raise RuntimeError("market data is empty")
    opens = raw["open_time_ms"].astype("int64")
    if not opens.is_monotonic_increasing or opens.duplicated().any():
        raise RuntimeError("market data ordering/duplicate failure")
    recent = opens.tail(min(120, len(opens))).tolist()
    if any(b - a != INTERVAL_MS for a, b in zip(recent, recent[1:])):
        raise RuntimeError("market data continuity failure: missing or irregular 1h candle")
    closed = raw[raw["close_time_ms"] < now_ms]
    if closed.empty:
        raise RuntimeError("market data has no completed candle")
    latest_close_ms = int(closed.iloc[-1]["close_time_ms"])
    if now_ms - latest_close_ms > 2 * INTERVAL_MS:
        raise RuntimeError("market data stale: latest completed candle is older than 2h")


def close_position(
    state: State,
    exit_time: pd.Timestamp,
    exit_price: float,
    reason: str,
    cfg: IchimokuKeltner1HConfig,
    notify: bool = True,
) -> dict:
    if state.position is None:
        raise RuntimeError("cannot close an empty position")
    p = state.position
    equity_before = state.equity_brl
    ret = net_return_pct(p["side"], float(p["entry_price"]), float(exit_price), cfg)
    pnl = equity_before * LEVERAGE * ret / 100.0
    state.equity_brl += pnl
    state.peak_equity_brl = max(state.peak_equity_brl, state.equity_brl)
    dd = (state.equity_brl / state.peak_equity_brl - 1.0) * 100.0
    state.max_drawdown_pct = min(state.max_drawdown_pct, dd)
    state.trade_count += 1
    record = {
        "trade_id": f"{p['side']}-{int(p['entry_open_ms'])}",
        "side": p["side"],
        "signal_time": p["signal_time"],
        "entry_time": p["entry_time"],
        "entry_price": float(p["entry_price"]),
        "entry_atr": float(p["entry_atr"]),
        "stop_price": float(p["stop_price"]),
        "exit_time": exit_time.isoformat(),
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "net_return_pct": ret,
        "pnl_brl_10x": pnl,
        "equity_before_brl": equity_before,
        "equity_after_brl": state.equity_brl,
    }
    state.position = None
    state.ledger_outbox.append(record)
    save_state(state)
    flush_ledger(state)
    if notify:
        queue_notification(
            state,
            f"exit-{record['trade_id']}",
            f"SAÍDA {p['side']}\nMotivo: {reason}\nSaída: {exit_price:.2f}\n"
            f"Retorno ativo líquido: {ret:+.3f}%\nResultado 10x: R$ {pnl:+,.2f}\n"
            f"Patrimônio: R$ {state.equity_brl:,.2f}\nDD atual: {dd:.2f}%\n"
            f"DD máximo: {state.max_drawdown_pct:.2f}%\nTrades: {state.trade_count}",
        )
    return record


def arm_signal(state: State, side: str, row: pd.Series, open_ms: int) -> None:
    atr = float(row["keltner_atr"])
    if pd.isna(atr):
        raise RuntimeError("signal has no completed ATR20")
    ts = pd.Timestamp(row["timestamp"], tz="UTC")
    state.pending_side = side
    state.pending_signal_time = ts.isoformat()
    state.pending_signal_open_ms = open_ms
    state.pending_atr = atr


def enter_pending(state: State, row: pd.Series, notify: bool = True) -> dict:
    if state.pending_side is None or state.pending_signal_open_ms is None or state.pending_atr is None:
        raise RuntimeError("incomplete pending signal state")
    open_ms = int(row["open_time_ms"])
    ts = pd.Timestamp(row["timestamp"], tz="UTC")
    p = open_position(
        state=state,
        side=state.pending_side,
        signal_time=str(state.pending_signal_time),
        signal_open_ms=int(state.pending_signal_open_ms),
        entry_time=ts,
        entry_open_ms=open_ms,
        entry_price=float(row["open"]),
        atr=float(state.pending_atr),
    )
    save_state(state)
    if notify:
        queue_notification(
            state,
            f"entry-{p['side']}-{p['entry_open_ms']}",
            f"ENTRADA {p['side']}\nETHUSDT 1H\nEntrada: {p['entry_price']:.2f}\nATR20: {p['entry_atr']:.2f}\n"
            f"Stop 2ATR: {p['stop_price']:.2f}\nExposição paper: R$ {state.equity_brl * LEVERAGE:,.2f}\n"
            f"Patrimônio: R$ {state.equity_brl:,.2f}",
        )
    return p


def target_for_row(side: str, row: pd.Series) -> float | None:
    value = row["executable_keltner_upper"] if side == "LONG" else row["executable_keltner_lower"]
    return None if pd.isna(value) else float(value)


def target_for_forming(side: str, latest_closed_indicator_row: pd.Series) -> float | None:
    value = latest_closed_indicator_row["keltner_upper"] if side == "LONG" else latest_closed_indicator_row["keltner_lower"]
    return None if pd.isna(value) else float(value)


def queue_daily_heartbeat(state: State, now: datetime, latest_market_open_ms: int) -> None:
    day = now.date().isoformat()
    if now.hour < HEARTBEAT_AFTER_UTC_HOUR or state.last_heartbeat_date == day:
        return
    position = state.position["side"] if state.position else "nenhuma"
    queue_notification(
        state,
        f"heartbeat-{day}",
        f"HEARTBEAT IA TRADER 💓\nStatus: operacional\nETHUSDT 1H\n"
        f"Patrimônio paper: R$ {state.equity_brl:,.2f}\nTrades: {state.trade_count}\n"
        f"Posição: {position}\nÚltima vela processada: {latest_market_open_ms}",
    )
    state.last_heartbeat_date = day
    save_state(state)


def write_health(
    state: State,
    status: str,
    now: datetime,
    latest_market_open_ms: int | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "status": status,
        "checked_at_utc": now.isoformat(),
        "last_success_utc": state.last_success_utc,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "latest_market_open_ms": latest_market_open_ms,
        "equity_brl": state.equity_brl,
        "trade_count": state.trade_count,
        "position": state.position["side"] if state.position else None,
        "pending_side": state.pending_side,
        "kill_switch_active": state.kill_switch_active,
        "notification_backlog": len(state.notification_outbox),
        "ledger_backlog": len(state.ledger_outbox),
        "error": error,
    }
    atomic_write_json(HEALTH_PATH, payload)


def execute_forward(state: State, now: datetime) -> int:
    now_ms = int(now.timestamp() * 1000)
    cfg = IchimokuKeltner1HConfig()
    assert_frozen_protocol(cfg)
    flush_ledger(state)
    reconcile_state_and_ledger(state)
    allow_new_entries = apply_control(state, load_control())

    raw = fetch_klines()
    validate_market_data(raw, now_ms)
    closed = raw[raw["close_time_ms"] < now_ms].copy().reset_index(drop=True)
    forming = raw[(raw["open_time_ms"] <= now_ms) & (raw["close_time_ms"] >= now_ms)].copy().reset_index(drop=True)
    data = enrich_indicators(closed, cfg).reset_index(drop=True)
    if data.empty:
        raise RuntimeError("indicator dataset is empty")

    latest_ms = int(closed.iloc[-1]["open_time_ms"])
    if not state.initialized:
        state.initialized = True
        state.last_processed_open_ms = latest_ms
        save_state(state)
        queue_notification(
            state,
            "forward-activation",
            "IA Trader ETH iniciado ✅\nForward test: ETHUSDT 1H\nCapital paper: R$ 500,00\n"
            "Exposição: 10x fixa\nStop: 2 ATR\nAlvo: Keltner 3.0\n"
            "Somente sinais observados após a ativação serão contabilizados.",
        )
        return latest_ms

    indices = [i for i in data.index if int(closed.iloc[i]["open_time_ms"]) > state.last_processed_open_ms]
    for i in indices:
        row = data.iloc[i]
        open_ms = int(closed.iloc[i]["open_time_ms"])
        ts = pd.Timestamp(row["timestamp"], tz="UTC")
        if ts < START_UTC:
            state.last_processed_open_ms = open_ms
            continue
        if ts > END_UTC:
            state.last_processed_open_ms = open_ms
            continue

        if allow_new_entries and state.position is None and state.pending_side is not None:
            expected_open_ms = next_open_ms(int(state.pending_signal_open_ms or 0))
            if open_ms == expected_open_ms:
                enter_pending(state, row)
            elif open_ms > expected_open_ms:
                raise RuntimeError("missed pending entry candle; refusing to fabricate a late fill")

        if state.position is not None and open_ms >= int(state.position["entry_open_ms"]):
            target = target_for_row(state.position["side"], row)
            exit_price, reason = evaluate_exit(state.position, row, target)
            if exit_price is not None and reason is not None:
                close_position(state, ts, exit_price, reason, cfg)

        if allow_new_entries and state.position is None and state.pending_side is None:
            if bool(row["long_signal"]):
                arm_signal(state, "LONG", row, open_ms)
            elif bool(row["short_signal"]):
                arm_signal(state, "SHORT", row, open_ms)

        state.last_processed_open_ms = open_ms
        save_state(state)

    if (
        allow_new_entries
        and not state.finalized
        and state.pending_side is not None
        and state.position is None
        and not forming.empty
    ):
        current = forming.iloc[-1]
        current_open_ms = int(current["open_time_ms"])
        current_ts = pd.Timestamp(current["timestamp"], tz="UTC")
        expected_open_ms = next_open_ms(int(state.pending_signal_open_ms or 0))
        if current_open_ms == expected_open_ms and current_ts <= END_UTC:
            enter_pending(state, current)

    if state.position is not None and not forming.empty:
        current = forming.iloc[-1]
        current_open_ms = int(current["open_time_ms"])
        if current_open_ms >= int(state.position["entry_open_ms"]):
            target = target_for_forming(state.position["side"], data.iloc[-1])
            exit_price, reason = evaluate_exit(state.position, current, target)
            if exit_price is not None and reason is not None:
                current_ts = pd.Timestamp(current["timestamp"], tz="UTC")
                close_position(state, current_ts, exit_price, reason, cfg)

    if pd.Timestamp(now) > END_UTC and not state.finalized:
        eligible = data[pd.to_datetime(data["timestamp"], utc=True) <= END_UTC]
        if state.position is not None and not eligible.empty:
            final_row = eligible.iloc[-1]
            final_ts = pd.Timestamp(final_row["timestamp"], tz="UTC")
            close_position(state, final_ts, float(final_row["close"]), "FORWARD_END_MTM", cfg)
        state.pending_side = None
        state.pending_signal_time = None
        state.pending_signal_open_ms = None
        state.pending_atr = None
        state.finalized = True
        save_state(state)
        queue_notification(
            state,
            "forward-finalized",
            f"FORWARD TEST ENCERRADO\nETHUSDT 1H\nPatrimônio final paper: R$ {state.equity_brl:,.2f}\n"
            f"Trades: {state.trade_count}\nDD máximo: {state.max_drawdown_pct:.2f}%",
        )

    flush_ledger(state)
    reconcile_state_and_ledger(state)
    save_state(state)
    return latest_ms


def main() -> None:
    state = load_state()
    now = utc_now()
    previous_status = state.ops_status
    latest_market_open_ms: int | None = None
    try:
        latest_market_open_ms = execute_forward(state, now)
        state.last_success_utc = now.isoformat()
        state.last_error_fingerprint = None
        if previous_status == "ERROR":
            queue_notification(
                state,
                f"ops-recovered-{now.date().isoformat()}",
                "IA Trader recuperado ✅\nDados, reconciliação e motor forward voltaram a executar normalmente.",
            )
        queue_daily_heartbeat(state, now, latest_market_open_ms)
        notifications_ok = flush_notifications(state)
        state.ops_status = "OK" if notifications_ok and not state.notification_outbox else "DEGRADED"
        save_state(state)
        write_health(state, state.ops_status, now, latest_market_open_ms)
    except Exception as exc:
        summary = f"{type(exc).__name__}: {str(exc)[:240]}"
        fingerprint = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16]
        if state.last_error_fingerprint != fingerprint:
            queue_notification(
                state,
                f"ops-error-{fingerprint}",
                f"ALERTA OPERACIONAL IA TRADER ⚠️\nForward em fail-closed.\n{summary}",
            )
        state.ops_status = "ERROR"
        state.last_error_fingerprint = fingerprint
        save_state(state)
        flush_notifications(state)
        write_health(state, "ERROR", now, latest_market_open_ms, summary)
        raise


if __name__ == "__main__":
    main()
