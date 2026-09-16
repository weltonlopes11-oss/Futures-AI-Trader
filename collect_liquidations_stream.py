from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import websocket


DEFAULT_BASE = "wss://fstream.binance.com/ws"


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def normalize_force_order(payload: dict, symbol: str) -> dict | None:
    data = payload.get("data", payload)
    order = data.get("o") if isinstance(data, dict) else None
    if not order or order.get("s") != symbol:
        return None

    event_ms = data.get("E") or order.get("T")
    if event_ms is None:
        observed = datetime.now(timezone.utc)
    else:
        observed = datetime.fromtimestamp(float(event_ms) / 1000.0, tz=timezone.utc)

    price = float(order.get("ap") or order.get("p") or 0.0)
    quantity = float(order.get("z") or order.get("q") or 0.0)
    if price <= 0 or quantity < 0:
        return None

    return {
        "timestamp": observed.isoformat(),
        "symbol": symbol,
        "side": str(order.get("S", "")).upper(),
        "order_type": order.get("o"),
        "status": order.get("X"),
        "price": price,
        "quantity": quantity,
        "notional": price * quantity,
        "trade_time_ms": order.get("T"),
    }


def main() -> None:
    symbol = os.getenv("LIQUIDATION_SYMBOL", "ETHUSDT").upper()
    base = os.getenv("BINANCE_FUTURES_WS_BASE", DEFAULT_BASE).rstrip("/")
    output = Path(
        os.getenv(
            "LIQUIDATION_OUTPUT",
            f"research_snapshots/liquidations/{symbol}_raw.csv",
        )
    )
    stream = f"{symbol.lower()}@forceOrder"
    url = f"{base}/{stream}"

    def on_open(ws):
        print(f"connected={url}")

    def on_message(ws, message):
        payload = json.loads(message)
        row = normalize_force_order(payload, symbol)
        if row is None:
            return
        append_row(output, row)
        direction = "LONG_LIQ" if row["side"] == "SELL" else "SHORT_LIQ"
        print(
            f"{row['timestamp']} {symbol} {direction} "
            f"notional={row['notional']:.2f}"
        )

    def on_error(ws, error):
        print(f"websocket_error={error}")

    def on_close(ws, status_code, msg):
        print(f"websocket_closed status={status_code} message={msg}")

    app = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    app.run_forever(ping_interval=120, ping_timeout=30)


if __name__ == "__main__":
    main()
