from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, metrics
from run_regime_discovery_2025_validate_2026 import run_featured_regime_window

ROBUST_REGIMES = {
    "LONG|LOW|CHOP",
    "SHORT|NORMAL|MIXED",
}

INITIAL_EQUITY_BRL = 2000.0
RISK_FRACTION = 0.10
MAX_LEVERAGE = 15.0

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
KLINE_PATH = "/fapi/v1/klines"

COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_binance_um_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    session: requests.Session,
) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list] = []
    cursor = start_ms

    while cursor < end_ms:
        response = session.get(
            BINANCE_FUTURES_BASE + KLINE_PATH,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1500,
            },
            timeout=60,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break

        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination cursor did not advance")
        cursor = next_cursor

        if len(batch) < 1500:
            break
        time.sleep(0.08)

    if not rows:
        raise RuntimeError(f"No Binance USD-M klines for {symbol} {interval}")

    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True).dt.tz_localize(None)

    for col in [
        "open", "high", "low", "close", "volume", "quote_volume",
        "taker_buy_base", "taker_buy_quote",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["trades"] = pd.to_numeric(frame["trades"], errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    start_naive = start.astimezone(timezone.utc).replace(tzinfo=None)
    end_naive = end.astimezone(timezone.utc).replace(tzinfo=None)
    frame = frame[(frame["timestamp"] >= start_naive) & (frame["timestamp"] < end_naive)].reset_index(drop=True)

    return frame


def apply_continuous_equity_sizing(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = trades.copy().sort_values("entry_time").reset_index(drop=True)
    equity = INITIAL_EQUITY_BRL
    peak = equity
    max_dd_pct = 0.0
    max_dd_brl = 0.0
    rows = []

    for idx, row in out.iterrows():
        entry = float(row["entry_price"])
        stop = float(row["stop_price"])
        stop_distance = abs(entry - stop) / entry
        if not np.isfinite(stop_distance) or stop_distance <= 0:
            raise ValueError(f"Invalid stop distance at row {idx}")

        leverage = min(MAX_LEVERAGE, RISK_FRACTION / stop_distance)
        equity_before = equity
        notional = equity_before * leverage
        pnl = notional * (float(row["net_return_pct"]) / 100.0)
        equity_after = equity_before + pnl

        peak = max(peak, equity_after)
        dd_brl = equity_after - peak
        dd_pct = (equity_after / peak - 1.0) * 100.0 if peak > 0 else float("nan")
        max_dd_brl = min(max_dd_brl, dd_brl)
        max_dd_pct = min(max_dd_pct, dd_pct)

        r = row.to_dict()
        r.update({
            "equity_before_brl": equity_before,
            "risk_budget_brl": equity_before * RISK_FRACTION,
            "stop_distance_pct": stop_distance * 100.0,
            "sizing_leverage": leverage,
            "notional_brl": notional,
            "trade_pnl_brl": pnl,
            "equity_after_brl": equity_after,
            "drawdown_pct_after_trade": dd_pct,
        })
        rows.append(r)
        equity = equity_after

    sized = pd.DataFrame(rows)

    summary = {
        "initial_equity_brl": INITIAL_EQUITY_BRL,
        "ending_equity_brl": float(equity),
        "total_pnl_brl": float(equity - INITIAL_EQUITY_BRL),
        "return_pct": float((equity / INITIAL_EQUITY_BRL - 1.0) * 100.0),
        "risk_fraction_of_current_equity": RISK_FRACTION,
        "max_leverage": MAX_LEVERAGE,
        "min_leverage_used": float(sized["sizing_leverage"].min()),
        "avg_leverage_used": float(sized["sizing_leverage"].mean()),
        "max_leverage_used": float(sized["sizing_leverage"].max()),
        "max_drawdown_brl": float(max_dd_brl),
        "max_drawdown_pct": float(max_dd_pct),
        "max_single_loss_brl": float(sized["trade_pnl_brl"].min()),
        "max_single_gain_brl": float(sized["trade_pnl_brl"].max()),
    }
    return sized, summary


def main() -> None:
    cfg = IchimokuKeltner1HConfig(
        fee_bps_per_side=4.0,
        slippage_bps_per_side=1.0,
    )

    eval_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    eval_end = datetime(2026, 9, 1, tzinfo=timezone.utc)
    warmup = eval_start - timedelta(days=120)

    with requests.Session() as session:
        h1 = fetch_binance_um_klines("ETHUSDT", "1h", warmup, eval_end, session)
        h4 = fetch_binance_um_klines("ETHUSDT", "4h", warmup - timedelta(days=15), eval_end, session)

    trades, counts = run_featured_regime_window(
        h1,
        h4,
        eval_start,
        eval_end,
        cfg,
        allowed_regimes=ROBUST_REGIMES,
    )

    sized, sizing = apply_continuous_equity_sizing(trades)

    result = {
        "experiment_id": "binance-direct-10pct-15x-2024-2026",
        "status": "research_only_not_promoted",
        "official_strategy_modified": False,
        "source": {
            "venue": "Binance USD-M Futures",
            "symbol": "ETHUSDT",
            "endpoint": "fapi.binance.com/fapi/v1/klines",
            "intervals": ["1h", "4h"],
            "direct_api": True,
            "csv_source": False,
            "data_vision_source": False,
            "h1_candles": int(len(h1)),
            "h4_candles": int(len(h4)),
            "h1_first": str(h1.iloc[0]["timestamp"]),
            "h1_last": str(h1.iloc[-1]["timestamp"]),
            "h4_first": str(h4.iloc[0]["timestamp"]),
            "h4_last": str(h4.iloc[-1]["timestamp"]),
        },
        "period": "2024-01-01 through 2026-08-31 UTC",
        "rules": {
            "allowed_regimes": sorted(ROBUST_REGIMES),
            "risk_fraction_of_current_equity": RISK_FRACTION,
            "max_leverage": MAX_LEVERAGE,
            "position_base": "100% of current equity",
            "stop": "2x ATR20",
            "target": "Keltner executable band",
            "fees_bps_per_side": 4.0,
            "slippage_bps_per_side": 1.0,
        },
        "trade_metrics_underlying": metrics(trades),
        "gate_counts": counts,
        "continuous_equity_sizing": sizing,
    }

    out = Path("artifacts/binance-direct-10pct-15x-2024-2026")
    out.mkdir(parents=True, exist_ok=True)
    sized.to_csv(out / "trades_binance_direct.csv", index=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
