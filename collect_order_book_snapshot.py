from __future__ import annotations

import csv
import os
from pathlib import Path

import requests

from backtest.order_book_liquidity import OrderBookLiquidity, OrderBookLiquidityConfig


BINANCE_FAPI_DEPTH = "https://fapi.binance.com/fapi/v1/depth"


def fetch_snapshot(symbol: str, limit: int = 100) -> dict:
    response = requests.get(
        BINANCE_FAPI_DEPTH,
        params={"symbol": symbol, "limit": limit},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    symbol = os.getenv("ORDER_BOOK_SYMBOL", "ETHUSDT").upper()
    api_limit = int(os.getenv("ORDER_BOOK_API_LIMIT", "100"))
    depth_levels = int(os.getenv("ORDER_BOOK_DEPTH_LEVELS", "50"))
    max_spread_bps = float(os.getenv("ORDER_BOOK_MAX_SPREAD_BPS", "5"))
    output = Path(
        os.getenv(
            "ORDER_BOOK_OUTPUT",
            f"research_snapshots/order_book/{symbol}.csv",
        )
    )

    snapshot = fetch_snapshot(symbol, api_limit)
    engine = OrderBookLiquidity(
        OrderBookLiquidityConfig(
            depth_levels=depth_levels,
            max_spread_bps=max_spread_bps,
        )
    )
    row = engine.from_snapshot(snapshot)
    row["timestamp"] = row["timestamp"].isoformat()
    row["symbol"] = symbol
    append_row(output, row)

    print(
        f"{symbol} order-book snapshot: "
        f"spread={row['spread_bps']:.4f} bps, "
        f"depth_imbalance={row['depth_imbalance']:.4f}, "
        f"microprice_edge={row['microprice_edge_bps']:.4f} bps, "
        f"signal={row['liquidity_signal']}"
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
