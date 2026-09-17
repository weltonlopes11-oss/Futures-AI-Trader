from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators

BINANCE_KLINE_ENDPOINTS = (
    "https://fapi.binance.com/fapi/v1/klines",
    "https://fapi1.binance.com/fapi/v1/klines",
    "https://fapi2.binance.com/fapi/v1/klines",
)
STATE_PATH = Path(os.getenv("FORWARD_STATE_PATH", "live/state.json"))
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
LEVERAGE = 10.0
INITIAL_EQUITY = 500.0
START_UTC = pd.Timestamp(os.getenv("FORWARD_START_UTC", "2026-09-17T00:00:00Z"))
END_UTC = pd.Timestamp(os.getenv("FORWARD_END_UTC", "2026-10-17T23:59:59Z"))


@dataclass
class State:
    equity_brl: float = INITIAL_EQUITY
    peak_equity_brl: float = INITIAL_EQUITY
    max_drawdown_pct: float = 0.0
    last_processed_open_ms: int = 0
    pending_side: str | None = None
    pending_signal_time: str | None = None
    position: dict | None = None
    trade_count: int = 0
    initialized: bool = False


def load_state() -> State:
    if not STATE_PATH.exists():
        return State()
    return State(**json.loads(STATE_PATH.read_text(encoding="utf-8")))


def save_state(state: State) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False), encoding="utf-8")


def telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body)
    with urllib.request.urlopen(req, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"Telegram HTTP {response.status}")


def fetch_klines(limit: int = 200) -> pd.DataFrame:
    query = urllib.parse.urlencode({"symbol": SYMBOL, "interval": INTERVAL, "limit": limit})
    errors: list[str] = []
    raw = None
    for endpoint in BINANCE_KLINE_ENDPOINTS:
        try:
            req = urllib.request.Request(
                f"{endpoint}?{query}",
                headers={"User-Agent": "Futures-AI-Trader/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                raw = json.loads(response.read())
            if not isinstance(raw, list):
                raise RuntimeError("unexpected Binance response")
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            code = getattr(exc, "code", "network")
            errors.append(f"{urllib.parse.urlparse(endpoint).netloc}:{code}")
    if raw is None:
        raise RuntimeError("All Binance Futures market-data endpoints failed: " + ", ".join(errors))

    rows = []
    for k in raw:
        rows.append({
            "open_time_ms": int(k[0]),
            "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True).tz_localize(None),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
            "close_time_ms": int(k[6]),
        })
    return pd.DataFrame(rows)


def net_return_pct(side: str, entry: float, exit_price: float, cfg: IchimokuKeltner1HConfig) -> float:
    gross = (exit_price / entry - 1.0) if side == "LONG" else (entry / exit_price - 1.0)
    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 10000.0
    return (gross - cost) * 100.0


def main() -> None:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cfg = IchimokuKeltner1HConfig()
    raw = fetch_klines()
    closed = raw[raw["close_time_ms"] < now_ms].copy().reset_index(drop=True)
    data = enrich_indicators(closed, cfg).reset_index(drop=True)
    state = load_state()
    if data.empty:
        return

    latest_ms = int(closed.iloc[-1]["open_time_ms"])
    if not state.initialized:
        state.initialized = True
        state.last_processed_open_ms = latest_ms
        save_state(state)
        telegram("IA Trader ETH iniciado ✅\nForward test: ETHUSDT 1H\nCapital paper: R$ 500,00\nExposição: 10x fixa\nStop: 2 ATR\nAlvo: Keltner 3.0\nSomente sinais observados após a ativação serão contabilizados.")
        return

    indices = [i for i in data.index if int(closed.iloc[i]["open_time_ms"]) > state.last_processed_open_ms]
    for i in indices:
        row = data.iloc[i]
        open_ms = int(closed.iloc[i]["open_time_ms"])
        ts = pd.Timestamp(row["timestamp"], tz="UTC")
        if ts < START_UTC or ts > END_UTC:
            state.last_processed_open_ms = open_ms
            continue

        if state.position is None and state.pending_side is not None:
            atr = float(data.iloc[i - 1]["keltner_atr"])
            entry = float(row["open"])
            stop = entry - 2.0 * atr if state.pending_side == "LONG" else entry + 2.0 * atr
            state.position = {"side": state.pending_side, "entry_time": ts.isoformat(), "entry_price": entry,
                              "entry_atr": atr, "stop_price": stop, "signal_time": state.pending_signal_time,
                              "entry_equity_brl": state.equity_brl}
            telegram(f"ENTRADA {state.pending_side}\nETHUSDT 1H\nEntrada: {entry:.2f}\nATR20: {atr:.2f}\nStop 2ATR: {stop:.2f}\nExposição paper: R$ {state.equity_brl*LEVERAGE:,.2f}\nPatrimônio: R$ {state.equity_brl:,.2f}")
            state.pending_side = None
            state.pending_signal_time = None

        if state.position is not None:
            p = state.position; side = p["side"]; stop = float(p["stop_price"])
            o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
            exit_price = None; reason = None
            if side == "LONG":
                target = row["executable_keltner_upper"]
                if l <= stop: exit_price, reason = min(o, stop), "STOP 2ATR"
                elif pd.notna(target) and h >= float(target): exit_price, reason = max(o, float(target)), "KELTNER SUPERIOR 3.0"
            else:
                target = row["executable_keltner_lower"]
                if h >= stop: exit_price, reason = max(o, stop), "STOP 2ATR"
                elif pd.notna(target) and l <= float(target): exit_price, reason = min(o, float(target)), "KELTNER INFERIOR 3.0"
            if exit_price is not None:
                ret = net_return_pct(side, float(p["entry_price"]), float(exit_price), cfg)
                pnl = state.equity_brl * LEVERAGE * ret / 100.0
                state.equity_brl += pnl
                state.peak_equity_brl = max(state.peak_equity_brl, state.equity_brl)
                dd = (state.equity_brl / state.peak_equity_brl - 1.0) * 100.0
                state.max_drawdown_pct = min(state.max_drawdown_pct, dd)
                state.trade_count += 1
                telegram(f"SAÍDA {side}\nMotivo: {reason}\nSaída: {exit_price:.2f}\nRetorno ativo líquido: {ret:+.3f}%\nResultado 10x: R$ {pnl:+,.2f}\nPatrimônio: R$ {state.equity_brl:,.2f}\nDD atual: {dd:.2f}%\nDD máximo: {state.max_drawdown_pct:.2f}%\nTrades: {state.trade_count}")
                state.position = None

        if state.position is None and state.pending_side is None:
            if bool(row["long_signal"]):
                state.pending_side, state.pending_signal_time = "LONG", ts.isoformat()
            elif bool(row["short_signal"]):
                state.pending_side, state.pending_signal_time = "SHORT", ts.isoformat()

        state.last_processed_open_ms = open_ms
        save_state(state)
    save_state(state)


if __name__ == "__main__":
    main()
