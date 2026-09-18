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

import numpy as np
import pandas as pd

from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators
from live.market_data import fetch_klines
from strategy.market_regime_gate import GateState, RegimeSnapshot, decide_regime

STATE_PATH = Path(os.getenv("FORWARD_STATE_PATH", "live/state.json"))
LEDGER_PATH = Path(os.getenv("FORWARD_LEDGER_PATH", "live/trades.csv"))
HEALTH_PATH = Path(os.getenv("FORWARD_HEALTH_PATH", "live/health.json"))
CONTROL_PATH = Path(os.getenv("FORWARD_CONTROL_PATH", "live/control.json"))
PROTOCOL_PATH = Path(
    os.getenv(
        "FROZEN_PROTOCOL_PATH",
        "research_protocol/official_regime_risk_v1_2026-09-18.json",
    )
)
EXPECTED_PROTOCOL_SHA256 = "4f1592c7ad5a1dc669d15d5f9618bf43f2a6d14f3a84275e3e3c8be879575a46"

SYMBOL = "ETHUSDT"
INTERVAL_MS = 60 * 60 * 1000
INITIAL_EQUITY = 2000.0
RISK_FRACTION = 0.10
MAX_LEVERAGE = 15.0
VOL_LOOKBACK_HOURS = 24 * 90
ER_PERIOD = 20
HEARTBEAT_AFTER_UTC_HOUR = 12
MAX_SENT_NOTIFICATION_IDS = 500

LEDGER_FIELDS = (
    "trade_id", "side", "regime_key", "signal_time", "entry_time",
    "entry_price", "entry_atr", "stop_price", "stop_distance_pct",
    "leverage", "notional_brl", "risk_budget_brl", "exit_time", "exit_price",
    "exit_reason", "net_return_pct", "pnl_brl", "equity_before_brl",
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
    pending_regime_key: str | None = None
    position: dict | None = None
    trade_count: int = 0
    initialized: bool = False
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
    return State(**{key: value for key, value in payload.items() if key in allowed})


def save_state(state: State) -> None:
    atomic_write_json(STATE_PATH, asdict(state))


def protocol_digest(path: Path = PROTOCOL_PATH) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def assert_frozen_protocol(cfg: IchimokuKeltner1HConfig) -> None:
    digest = protocol_digest()
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"official protocol integrity failure: {digest}")
    if cfg != IchimokuKeltner1HConfig():
        raise RuntimeError("runtime Ichimoku/Keltner config differs from official protocol")


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
    ok = True
    for event_id, text in list(state.notification_outbox.items()):
        try:
            telegram(text)
        except Exception:
            ok = False
            continue
        state.notification_outbox.pop(event_id, None)
        state.sent_notification_ids.append(event_id)
        state.sent_notification_ids = state.sent_notification_ids[-MAX_SENT_NOTIFICATION_IDS:]
        save_state(state)
    return ok


def load_control() -> dict:
    if not CONTROL_PATH.exists():
        return {"kill_switch": False, "reason": ""}
    payload = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload.get("kill_switch"), bool):
        raise RuntimeError("invalid control: kill_switch must be boolean")
    return {"kill_switch": payload["kill_switch"], "reason": str(payload.get("reason", "")).strip()}


def apply_control(state: State, control: dict) -> bool:
    active = bool(control["kill_switch"])
    if active and not state.kill_switch_active:
        state.pending_side = None
        state.pending_signal_time = None
        state.pending_signal_open_ms = None
        state.pending_atr = None
        state.pending_regime_key = None
        state.kill_switch_active = True
        queue_notification(state, "kill-switch-on", "KILL SWITCH ATIVO 🛑\nNovas entradas bloqueadas.")
    elif not active and state.kill_switch_active:
        state.kill_switch_active = False
        queue_notification(state, f"kill-switch-off-{utc_now().isoformat()}", "KILL SWITCH LIBERADO ✅")
    return not active


def add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("timestamp").reset_index(drop=True)
    close = out["close"].astype(float)
    out["atr_price"] = out["keltner_atr"].astype(float) / close.replace(0.0, np.nan)
    minp = 24 * 30
    out["atr_price_q33"] = out["atr_price"].rolling(VOL_LOOKBACK_HOURS, min_periods=minp).quantile(0.33)
    out["atr_price_q67"] = out["atr_price"].rolling(VOL_LOOKBACK_HOURS, min_periods=minp).quantile(0.67)
    direction = (close - close.shift(ER_PERIOD)).abs()
    path = close.diff().abs().rolling(ER_PERIOD, min_periods=ER_PERIOD).sum()
    out["er20"] = direction / path.replace(0.0, np.nan)
    return out


def build_4h_regime(h4: pd.DataFrame) -> pd.DataFrame:
    out = h4.copy().sort_values("timestamp").reset_index(drop=True)
    out["ema50_4h"] = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema50_slope"] = out["ema50_4h"] - out["ema50_4h"].shift(1)
    bullish = (out["close"] > out["ema50_4h"]) & (out["ema50_slope"] > 0)
    bearish = (out["close"] < out["ema50_4h"]) & (out["ema50_slope"] < 0)
    out["regime_4h"] = np.select([bullish, bearish], ["BULLISH", "BEARISH"], default="NEUTRAL")
    out["regime_available_time"] = pd.to_datetime(out["timestamp"]) + pd.Timedelta(hours=4)
    return out[["regime_available_time", "regime_4h", "ema50_4h", "ema50_slope"]].dropna()


def attach_4h_regime(h1: pd.DataFrame, h4: pd.DataFrame) -> pd.DataFrame:
    one = h1.copy().sort_values("timestamp").reset_index(drop=True)
    one["signal_available_time"] = pd.to_datetime(one["timestamp"]) + pd.Timedelta(hours=1)
    regime = build_4h_regime(h4).sort_values("regime_available_time")
    return pd.merge_asof(
        one.sort_values("signal_available_time"),
        regime,
        left_on="signal_available_time",
        right_on="regime_available_time",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("timestamp").reset_index(drop=True)


def enrich_live(h1_closed: pd.DataFrame, h4_closed: pd.DataFrame, cfg: IchimokuKeltner1HConfig) -> pd.DataFrame:
    return attach_4h_regime(add_regime_features(enrich_indicators(h1_closed, cfg)), h4_closed)


def validate_market_data(raw: pd.DataFrame, now_ms: int, interval_ms: int) -> None:
    if raw.empty:
        raise RuntimeError("market data is empty")
    opens = raw["open_time_ms"].astype("int64")
    if not opens.is_monotonic_increasing or opens.duplicated().any():
        raise RuntimeError("market data ordering/duplicate failure")
    recent = opens.tail(min(120, len(opens))).tolist()
    if any(b - a != interval_ms for a, b in zip(recent, recent[1:])):
        raise RuntimeError("market data continuity failure")
    closed = raw[raw["close_time_ms"] < now_ms]
    if closed.empty:
        raise RuntimeError("market data has no completed candle")


def net_return_pct(side: str, entry: float, exit_price: float, cfg: IchimokuKeltner1HConfig) -> float:
    gross = (exit_price / entry - 1.0) if side == "LONG" else (entry / exit_price - 1.0)
    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 10000.0
    return (gross - cost) * 100.0


def next_open_ms(signal_open_ms: int) -> int:
    return signal_open_ms + INTERVAL_MS


def size_position(equity: float, entry: float, stop: float) -> tuple[float, float, float, float]:
    distance = abs(entry - stop) / entry
    if distance <= 0 or not math.isfinite(distance):
        raise RuntimeError("invalid stop distance")
    leverage = min(MAX_LEVERAGE, RISK_FRACTION / distance)
    notional = equity * leverage
    risk_budget = equity * RISK_FRACTION
    return distance, leverage, notional, risk_budget


def open_position(state: State, row: pd.Series) -> dict:
    if state.pending_side is None or state.pending_signal_open_ms is None or state.pending_atr is None:
        raise RuntimeError("incomplete pending signal")
    open_ms = int(row["open_time_ms"])
    if open_ms != next_open_ms(int(state.pending_signal_open_ms)):
        raise RuntimeError("entry timing invariant failed")
    side = state.pending_side
    entry = float(row["open"])
    atr = float(state.pending_atr)
    stop = entry - 2.0 * atr if side == "LONG" else entry + 2.0 * atr
    distance, leverage, notional, risk_budget = size_position(state.equity_brl, entry, stop)
    p = {
        "side": side,
        "regime_key": state.pending_regime_key,
        "signal_time": state.pending_signal_time,
        "signal_open_ms": state.pending_signal_open_ms,
        "entry_time": pd.Timestamp(row["timestamp"], tz="UTC").isoformat(),
        "entry_open_ms": open_ms,
        "entry_price": entry,
        "entry_atr": atr,
        "stop_price": stop,
        "stop_distance_pct": distance * 100.0,
        "leverage": leverage,
        "notional_brl": notional,
        "risk_budget_brl": risk_budget,
        "entry_equity_brl": state.equity_brl,
    }
    state.position = p
    state.pending_side = None
    state.pending_signal_time = None
    state.pending_signal_open_ms = None
    state.pending_atr = None
    state.pending_regime_key = None
    save_state(state)
    queue_notification(
        state,
        f"entry-{side}-{open_ms}",
        f"ENTRADA OFICIAL {side} ✅\nETHUSDT 1H\nRegime: {p['regime_key']}\n"
        f"Entrada: {entry:.2f}\nStop 2ATR: {stop:.2f}\n"
        f"Alavancagem dinâmica: {leverage:.2f}x\nNotional paper: R$ {notional:,.2f}\n"
        f"Risco no stop: ~R$ {risk_budget:,.2f} (10%)\nPatrimônio paper: R$ {state.equity_brl:,.2f}",
    )
    return p


def evaluate_exit(position: dict, row: pd.Series, target: float | None) -> tuple[float | None, str | None]:
    side = position["side"]
    stop = float(position["stop_price"])
    o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
    if side == "LONG":
        if l <= stop:
            return min(o, stop), "STOP 2ATR"
        if target is not None and h >= target:
            return max(o, target), "KELTNER SUPERIOR 3.0"
    else:
        if h >= stop:
            return max(o, stop), "STOP 2ATR"
        if target is not None and l <= target:
            return min(o, target), "KELTNER INFERIOR 3.0"
    return None, None


def ledger_records() -> list[dict]:
    if not LEDGER_PATH.exists() or LEDGER_PATH.stat().st_size == 0:
        return []
    with LEDGER_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_trade(record: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if any(row.get("trade_id") == record["trade_id"] for row in ledger_records()):
        return
    exists = LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0
    with LEDGER_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in LEDGER_FIELDS})


def flush_ledger(state: State) -> None:
    for record in list(state.ledger_outbox):
        append_trade(record)
        state.ledger_outbox.remove(record)
        save_state(state)


def close_position(state: State, ts: pd.Timestamp, exit_price: float, reason: str, cfg: IchimokuKeltner1HConfig) -> None:
    if state.position is None:
        return
    p = state.position
    ret = net_return_pct(p["side"], float(p["entry_price"]), exit_price, cfg)
    equity_before = state.equity_brl
    pnl = float(p["notional_brl"]) * ret / 100.0
    state.equity_brl = equity_before + pnl
    state.peak_equity_brl = max(state.peak_equity_brl, state.equity_brl)
    dd = (state.equity_brl / state.peak_equity_brl - 1.0) * 100.0
    state.max_drawdown_pct = min(state.max_drawdown_pct, dd)
    state.trade_count += 1
    record = {
        "trade_id": f"{p['side']}-{int(p['entry_open_ms'])}",
        "side": p["side"], "regime_key": p["regime_key"],
        "signal_time": p["signal_time"], "entry_time": p["entry_time"],
        "entry_price": p["entry_price"], "entry_atr": p["entry_atr"],
        "stop_price": p["stop_price"], "stop_distance_pct": p["stop_distance_pct"],
        "leverage": p["leverage"], "notional_brl": p["notional_brl"],
        "risk_budget_brl": p["risk_budget_brl"], "exit_time": ts.isoformat(),
        "exit_price": exit_price, "exit_reason": reason, "net_return_pct": ret,
        "pnl_brl": pnl, "equity_before_brl": equity_before,
        "equity_after_brl": state.equity_brl,
    }
    state.position = None
    state.ledger_outbox.append(record)
    save_state(state)
    flush_ledger(state)
    queue_notification(
        state,
        f"exit-{record['trade_id']}",
        f"SAÍDA OFICIAL {p['side']}\nMotivo: {reason}\nSaída: {exit_price:.2f}\n"
        f"Retorno líquido do ativo: {ret:+.3f}%\nAlavancagem usada: {p['leverage']:.2f}x\n"
        f"Resultado paper: R$ {pnl:+,.2f}\nPatrimônio: R$ {state.equity_brl:,.2f}\n"
        f"DD atual: {dd:.2f}% | DD máximo: {state.max_drawdown_pct:.2f}%\nTrades: {state.trade_count}",
    )


def arm_if_authorized(state: State, row: pd.Series, open_ms: int) -> None:
    side = "LONG" if bool(row["long_signal"]) else "SHORT" if bool(row["short_signal"]) else None
    if side is None:
        return
    needed = ["atr_price", "atr_price_q33", "atr_price_q67", "er20", "regime_4h", "keltner_atr"]
    if any(pd.isna(row.get(k)) for k in needed):
        return
    snapshot = RegimeSnapshot(
        side=side,
        regime_4h=str(row["regime_4h"]),
        atr_price=float(row["atr_price"]),
        atr_price_q33=float(row["atr_price_q33"]),
        atr_price_q67=float(row["atr_price_q67"]),
        er20=float(row["er20"]),
    )
    decision = decide_regime(snapshot)
    if decision.state is not GateState.ON:
        return
    state.pending_side = side
    state.pending_signal_time = pd.Timestamp(row["timestamp"], tz="UTC").isoformat()
    state.pending_signal_open_ms = open_ms
    state.pending_atr = float(row["keltner_atr"])
    state.pending_regime_key = decision.regime_key
    save_state(state)
    queue_notification(
        state,
        f"signal-{side}-{open_ms}",
        f"SINAL AUTORIZADO {side} 🟢\nETHUSDT 1H\nRegime: {decision.regime_key}\n"
        f"Entrada programada: abertura da próxima vela 1H\nATR20: {float(row['keltner_atr']):.2f}",
    )


def target_for_row(side: str, row: pd.Series) -> float | None:
    key = "executable_keltner_upper" if side == "LONG" else "executable_keltner_lower"
    value = row[key]
    return None if pd.isna(value) else float(value)


def target_for_forming(side: str, latest_closed: pd.Series) -> float | None:
    key = "keltner_upper" if side == "LONG" else "keltner_lower"
    value = latest_closed[key]
    return None if pd.isna(value) else float(value)


def queue_daily_heartbeat(state: State, now: datetime, latest_ms: int) -> None:
    day = now.date().isoformat()
    if now.hour < HEARTBEAT_AFTER_UTC_HOUR or state.last_heartbeat_date == day:
        return
    position = state.position["side"] if state.position else "nenhuma"
    queue_notification(
        state,
        f"heartbeat-{day}",
        f"HEARTBEAT IA TRADER 💓\nEstratégia: OFFICIAL REGIME RISK V1\n"
        f"Patrimônio paper: R$ {state.equity_brl:,.2f}\nTrades: {state.trade_count}\n"
        f"Posição: {position}\nÚltima vela 1H: {latest_ms}",
    )
    state.last_heartbeat_date = day


def write_health(state: State, status: str, now: datetime, latest_ms: int | None = None, error: str | None = None) -> None:
    atomic_write_json(HEALTH_PATH, {
        "status": status,
        "strategy": "official_regime_risk_v1",
        "checked_at_utc": now.isoformat(),
        "last_success_utc": state.last_success_utc,
        "symbol": SYMBOL,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "latest_market_open_ms": latest_ms,
        "equity_brl": state.equity_brl,
        "trade_count": state.trade_count,
        "position": state.position["side"] if state.position else None,
        "pending_side": state.pending_side,
        "kill_switch_active": state.kill_switch_active,
        "error": error,
    })


def execute_forward(state: State, now: datetime) -> int:
    now_ms = int(now.timestamp() * 1000)
    cfg = IchimokuKeltner1HConfig()
    assert_frozen_protocol(cfg)
    flush_ledger(state)
    allow_new_entries = apply_control(state, load_control())

    raw1 = fetch_klines("1h", 2500)
    raw4 = fetch_klines("4h", 400)
    validate_market_data(raw1, now_ms, INTERVAL_MS)
    validate_market_data(raw4, now_ms, 4 * INTERVAL_MS)

    closed1 = raw1[raw1["close_time_ms"] < now_ms].copy().reset_index(drop=True)
    closed4 = raw4[raw4["close_time_ms"] < now_ms].copy().reset_index(drop=True)
    forming1 = raw1[(raw1["open_time_ms"] <= now_ms) & (raw1["close_time_ms"] >= now_ms)].copy().reset_index(drop=True)
    data = enrich_live(closed1, closed4, cfg)
    if data.empty:
        raise RuntimeError("indicator dataset empty")

    latest_ms = int(closed1.iloc[-1]["open_time_ms"])
    if not state.initialized:
        state.initialized = True
        state.last_processed_open_ms = latest_ms
        save_state(state)
        queue_notification(
            state,
            "official-regime-risk-v1-activation",
            "IA TRADER OFICIAL ATIVADO ✅\nETHUSDT Binance USD-M Futures\n"
            "Regimes: LONG|LOW|CHOP e SHORT|NORMAL|MIXED\n"
            "Risk sizing: 10% do patrimônio no stop | teto 15x\n"
            "Capital paper inicial: R$ 2.000,00\n"
            "Somente novos sinais após esta ativação serão contabilizados.",
        )
        return latest_ms

    indices = [i for i in data.index if int(closed1.iloc[i]["open_time_ms"]) > state.last_processed_open_ms]
    for i in indices:
        row = data.iloc[i]
        open_ms = int(closed1.iloc[i]["open_time_ms"])
        ts = pd.Timestamp(row["timestamp"], tz="UTC")

        if allow_new_entries and state.position is None and state.pending_side is not None:
            expected = next_open_ms(int(state.pending_signal_open_ms or 0))
            if open_ms == expected:
                open_position(state, row)
            elif open_ms > expected:
                raise RuntimeError("missed pending entry candle")

        if state.position is not None and open_ms >= int(state.position["entry_open_ms"]):
            target = target_for_row(state.position["side"], row)
            exit_price, reason = evaluate_exit(state.position, row, target)
            if exit_price is not None:
                close_position(state, ts, exit_price, str(reason), cfg)

        if allow_new_entries and state.position is None and state.pending_side is None:
            arm_if_authorized(state, row, open_ms)

        state.last_processed_open_ms = open_ms
        save_state(state)

    if allow_new_entries and state.position is None and state.pending_side is not None and not forming1.empty:
        current = forming1.iloc[-1]
        if int(current["open_time_ms"]) == next_open_ms(int(state.pending_signal_open_ms or 0)):
            open_position(state, current)

    if state.position is not None and not forming1.empty:
        current = forming1.iloc[-1]
        target = target_for_forming(state.position["side"], data.iloc[-1])
        exit_price, reason = evaluate_exit(state.position, current, target)
        if exit_price is not None:
            close_position(state, pd.Timestamp(current["timestamp"], tz="UTC"), exit_price, str(reason), cfg)

    flush_ledger(state)
    save_state(state)
    return latest_ms


def main() -> None:
    state = load_state()
    now = utc_now()
    latest_ms: int | None = None
    previous_status = state.ops_status
    try:
        latest_ms = execute_forward(state, now)
        state.last_success_utc = now.isoformat()
        state.last_error_fingerprint = None
        if previous_status == "ERROR":
            queue_notification(state, f"recovered-{now.isoformat()}", "IA Trader recuperado ✅")
        queue_daily_heartbeat(state, now, latest_ms)
        notifications_ok = flush_notifications(state)
        state.ops_status = "OK" if notifications_ok and not state.notification_outbox else "DEGRADED"
        save_state(state)
        write_health(state, state.ops_status, now, latest_ms)
    except Exception as exc:
        summary = f"{type(exc).__name__}: {str(exc)[:240]}"
        fingerprint = hashlib.sha256(summary.encode()).hexdigest()[:16]
        if state.last_error_fingerprint != fingerprint:
            queue_notification(state, f"error-{fingerprint}", f"ALERTA IA TRADER ⚠️\nFail-closed.\n{summary}")
        state.ops_status = "ERROR"
        state.last_error_fingerprint = fingerprint
        save_state(state)
        flush_notifications(state)
        write_health(state, "ERROR", now, latest_ms, summary)
        raise


if __name__ == "__main__":
    main()
