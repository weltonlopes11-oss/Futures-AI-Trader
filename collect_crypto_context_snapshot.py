from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from backtest.binance_positioning_context import BinancePositioningContextLoader
from backtest.cross_exchange_derivatives import CrossExchangeDerivatives


def parse_utc(name: str, default: str) -> datetime:
    value = os.getenv(name, default).replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main() -> None:
    symbol = os.getenv("CRYPTO_CONTEXT_SYMBOL", "ETHUSDT").upper()
    start = parse_utc("CRYPTO_CONTEXT_START_UTC", "2026-09-01T00:00:00Z")
    end = parse_utc("CRYPTO_CONTEXT_END_UTC", "2026-09-15T00:00:00Z")
    out = Path(os.getenv("CRYPTO_CONTEXT_OUTPUT_DIR", "research_snapshots/crypto_context"))
    out.mkdir(parents=True, exist_ok=True)

    loader = BinancePositioningContextLoader()
    datasets = {
        "premium_15m": loader.fetch_premium_index(symbol, start, end, interval="15m"),
        "global_ls_1h": loader.fetch_global_ratio(symbol, start, end),
        "top_account_ls_1h": loader.fetch_top_account_ratio(symbol, start, end),
        "top_position_ls_1h": loader.fetch_top_position_ratio(symbol, start, end),
    }
    for name, frame in datasets.items():
        path = out / f"{symbol}_{name}_{start.date()}_{end.date()}.csv"
        frame.to_csv(path, index=False)
        print(f"saved={path} rows={len(frame)}")

    cross = CrossExchangeDerivatives()
    live = cross.snapshot()
    live_path = out / f"{symbol}_cross_exchange_live.csv"
    live.to_csv(live_path, index=False)
    print(f"saved={live_path} rows={len(live)} errors={live.attrs.get('errors', [])}")
    print(cross.aggregate(live))


if __name__ == "__main__":
    main()
