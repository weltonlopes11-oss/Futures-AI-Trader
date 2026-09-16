from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.operational_backtest import OperationalBacktest


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def trend(frame: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    fast_ma = frame["close"].ewm(span=fast, adjust=False).mean()
    slow_ma = frame["close"].ewm(span=slow, adjust=False).mean()
    return pd.Series(pd.NA, index=frame.index).mask(fast_ma > slow_ma, "LONG").mask(fast_ma < slow_ma, "SHORT")


def causal_context(signal: pd.DataFrame, higher: pd.DataFrame, label: str) -> pd.DataFrame:
    h = higher.copy()
    h[label] = trend(h, 12, 26)
    # Data Vision loader already normalizes close_time to UTC-naive datetime.
    h["available_at"] = pd.to_datetime(h["close_time"], errors="coerce")
    return pd.merge_asof(
        signal.sort_values("timestamp"),
        h[["available_at", label]].dropna().sort_values("available_at"),
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    )


def require_coverage(frame: pd.DataFrame, expected: int, label: str) -> None:
    minimum = int(expected * 0.99)
    if len(frame) < minimum:
        raise RuntimeError(f"Insufficient {label} coverage: got {len(frame)}, expected at least {minimum} of {expected}")


def main():
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    start = datetime.fromisoformat(os.getenv("BACKTEST_START_UTC", "2026-09-01T00:00:00+00:00").replace("Z", "+00:00"))
    end = datetime.fromisoformat(os.getenv("BACKTEST_END_UTC", "2026-09-15T00:00:00+00:00").replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    loader = BinanceDataVisionLoader()
    m15 = loader.fetch_window(symbol, "15m", start, end)
    h1 = loader.fetch_window(symbol, "1h", start, end)
    h4 = loader.fetch_window(symbol, "4h", start, end)
    total_minutes = int((end - start).total_seconds() // 60)
    require_coverage(m15, total_minutes // 15, "15m")
    require_coverage(h1, total_minutes // 60, "1h")
    require_coverage(h4, total_minutes // 240, "4h")

    m15["atr"] = atr(m15)
    m15["signal_15m"] = trend(m15, 12, 26)
    context = causal_context(m15, h1, "trend_1h")
    context = causal_context(context, h4, "trend_4h")
    context["decision"] = "NO_TRADE"
    long_mask = (context["signal_15m"] == "LONG") & (context["trend_1h"] == "LONG") & (context["trend_4h"] == "LONG")
    short_mask = (context["signal_15m"] == "SHORT") & (context["trend_1h"] == "SHORT") & (context["trend_4h"] == "SHORT")
    context.loc[long_mask, "decision"] = "LONG"
    context.loc[short_mask, "decision"] = "SHORT"

    engine = OperationalBacktest(
        fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")),
        slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")),
    )
    out = Path("artifacts/operational-benchmark")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rr in (1.0, 1.5, 2.0, 3.0):
        trades = engine.run(context, context[["timestamp", "decision"]], rr=rr)
        metrics = engine.metrics(trades)
        metrics["rr"] = rr
        rows.append(metrics)
        trades.to_csv(out / f"trades_rr_{rr}.csv", index=False)
    results = pd.DataFrame(rows)[["rr", "trades", "win_rate_pct", "net_return_pct", "profit_factor", "expectancy_pct", "max_drawdown_pct", "payoff", "long", "short"]]
    results.to_csv(out / "results.csv", index=False)
    manifest = {"source":"Binance Data Vision","market":"USD-M Futures","symbol":symbol,"start_utc":start.isoformat(),"end_utc":end.isoformat(),"signal":"15m","structure":"1h","regime":"4h","fee_bps_per_side":engine.fee_bps_per_side,"slippage_bps_per_side":engine.slippage_bps_per_side,"candles_15m":len(m15),"candles_1h":len(h1),"candles_4h":len(h4)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
