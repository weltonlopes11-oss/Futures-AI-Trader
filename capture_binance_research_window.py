from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from backtest.binance_historical_collector import BinanceHistoricalDataCollector, BinanceWindow


def parse_utc(name: str, default: datetime) -> datetime:
    value = os.getenv(name)
    if not value:
        return default
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main():
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    default_end = now - timedelta(minutes=15)
    default_start = default_end - timedelta(days=14)
    start = parse_utc("BACKTEST_START_UTC", default_start)
    end = parse_utc("BACKTEST_END_UTC", default_end)
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")

    collector = BinanceHistoricalDataCollector()
    frames = collector.collect(BinanceWindow(symbol=symbol, start_time=start, end_time=end))

    out = Path("artifacts") / "binance-window"
    out.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, frame in frames.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    signal = frames["signal_15m"]
    oi = frames["open_interest"]
    manifest = {
        "source": "Binance",
        "market": "USD-M Futures",
        "symbol": symbol,
        "start_utc": start.astimezone(timezone.utc).isoformat(),
        "end_utc": end.astimezone(timezone.utc).isoformat(),
        "signal_interval": "15m",
        "structure_interval": "1h",
        "regime_interval": "4h",
        "oi_period": "15m",
        "signal_candles": len(signal),
        "structure_1h_candles": len(frames["structure_1h"]),
        "regime_4h_candles": len(frames["regime_4h"]),
        "oi_observations": len(oi),
        "oi_coverage_pct": float(signal["open_interest"].notna().mean() * 100.0) if len(signal) else 0.0,
        "checksums_sha256": checksums,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
