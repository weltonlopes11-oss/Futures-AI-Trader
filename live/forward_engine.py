from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators
from live.market_data import fetch_klines

STATE_PATH = Path(os.getenv("FORWARD_STATE_PATH", "live/state.json"))
LEDGER_PATH = Path(os.getenv("FORWARD_LEDGER_PATH", "live/trades.csv"))
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


def load_state() -> State:
    if not STATE_PATH.exists():
        return State()
    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    allowed = State.__dataclass_fields__.keys()
    return State(**{key: value for key, value in payload.items() if key in allowed})


def save_state(state: State) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False), encoding="utf-8")


def protocol_digest(path: Path = PROTOCOL_PATH) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def assert_frozen_protocol(cfg: IchimokuKeltner1HConfig, path: Path = PROTOCOL_PATH) -> None:
    digest = protocol_digest(path)
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"frozen protocol integrity failure: {digest}")
    expected = IchimokuKeltner1HConfig()
    if cfg != expected:
        raise RuntimeError("runtime strategy config differs from frozen protocol")


def telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body)
    with urllib.request.urlopen(req, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"Telegram HTTP {response.status}")


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


def append_trade(record: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("r", encoding="utf-8", newline="") as handle:
            if any(row.get("trade_id") == record["trade_id"] for row in csv.DictReader(handle)):
                return
    exists = LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0
    with LEDGER_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in LEDGER_FIELDS})


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
    append_trade(record)
    state.position = None
    if notify:
        telegram(
            f"SAÍDA {p['side']}\nMotivo: {reason}\nSaída: {exit_price:.2f}\n"
            f"Retorno ativo líquido: {ret:+.3f}%\nResultado 10x: R$ {pnl:+,.2f}\n"
            f"Patrimônio: R$ {state.equity_brl:,.2f}\nDD atual: {dd:.2f}%\n"
            f"DD máximo: {state.max_drawdown_pct:.2f}%\nTrades: {state.trade_count}"
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
    if notify:
        telegram(
            f"ENTRADA {p['side']}\nETHUSDT 1H\nEntrada: {p['entry_price']:.2f}\nATR20: {p['entry_atr']:.2f}\n"
            f"Stop 2ATR: {p['stop_price']:.2f}\nExposição paper: R$ {state.equity_brl * LEVERAGE:,.2f}\n"
            f"Patrimônio: R$ {state.equity_brl:,.2f}"
        )
    return p


def target_for_row(side: str, row: pd.Series) -> float | None:
    value = row["executable_keltner_upper"] if side == "LONG" else row["executable_keltner_lower"]
    return None if pd.isna(value) else float(value)


def target_for_forming(side: str, latest_closed_indicator_row: pd.Series) -> float | None:
    value = latest_closed_indicator_row["keltner_upper"] if side == "LONG" else latest_closed_indicator_row["keltner_lower"]
    return None if pd.isna(value) else float(value)


def main() -> None:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cfg = IchimokuKeltner1HConfig()
    assert_frozen_protocol(cfg)
    raw = fetch_klines()
    closed = raw[raw["close_time_ms"] < now_ms].copy().reset_index(drop=True)
    forming = raw[(raw["open_time_ms"] <= now_ms) & (raw["close_time_ms"] >= now_ms)].copy().reset_index(drop=True)
    data = enrich_indicators(closed, cfg).reset_index(drop=True)
    state = load_state()
    if data.empty:
        return

    latest_ms = int(closed.iloc[-1]["open_time_ms"])
    if not state.initialized:
        state.initialized = True
        state.last_processed_open_ms = latest_ms
        save_state(state)
        telegram(
            "IA Trader ETH iniciado ✅\nForward test: ETHUSDT 1H\nCapital paper: R$ 500,00\n"
            "Exposição: 10x fixa\nStop: 2 ATR\nAlvo: Keltner 3.0\n"
            "Somente sinais observados após a ativação serão contabilizados."
        )
        return

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

        if state.position is None and state.pending_side is not None:
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

        if state.position is None and state.pending_side is None:
            if bool(row["long_signal"]):
                arm_signal(state, "LONG", row, open_ms)
            elif bool(row["short_signal"]):
                arm_signal(state, "SHORT", row, open_ms)

        state.last_processed_open_ms = open_ms
        save_state(state)

    # The signal on completed candle t is now known. Execute immediately at the
    # already-known open of the currently-forming t+1 candle instead of waiting
    # until t+1 closes one hour later.
    if not state.finalized and state.pending_side is not None and state.position is None and not forming.empty:
        current = forming.iloc[-1]
        current_open_ms = int(current["open_time_ms"])
        current_ts = pd.Timestamp(current["timestamp"], tz="UTC")
        expected_open_ms = next_open_ms(int(state.pending_signal_open_ms or 0))
        if current_open_ms == expected_open_ms and current_ts <= END_UTC:
            enter_pending(state, current)

    # A position can touch stop/target during the first minutes of the forming
    # candle. Inspect only market data observable at this run; the same candle
    # is re-evaluated when closed on the next run if no exit has happened yet.
    if state.position is not None and not forming.empty:
        current = forming.iloc[-1]
        current_open_ms = int(current["open_time_ms"])
        if current_open_ms >= int(state.position["entry_open_ms"]):
            target = target_for_forming(state.position["side"], data.iloc[-1])
            exit_price, reason = evaluate_exit(state.position, current, target)
            if exit_price is not None and reason is not None:
                current_ts = pd.Timestamp(current["timestamp"], tz="UTC")
                close_position(state, current_ts, exit_price, reason, cfg)

    # Close any remaining paper position at the evaluation boundary. No new
    # entries are permitted after END_UTC and finalization is idempotent.
    if pd.Timestamp.now(tz="UTC") > END_UTC and not state.finalized:
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
        telegram(
            f"FORWARD TEST ENCERRADO\nETHUSDT 1H\nPatrimônio final paper: R$ {state.equity_brl:,.2f}\n"
            f"Trades: {state.trade_count}\nDD máximo: {state.max_drawdown_pct:.2f}%"
        )

    save_state(state)


if __name__ == "__main__":
    main()
