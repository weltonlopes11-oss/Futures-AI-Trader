from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators, metrics


def run_window(data: pd.DataFrame, eval_start: datetime, eval_end: datetime, cfg: IchimokuKeltner1HConfig) -> pd.DataFrame:
    enriched = enrich_indicators(data, cfg)
    start = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    trades = []
    position = None
    pending = None
    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 100.0

    for i, row in enriched.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        in_eval = start <= ts < end

        if position is None and pending is not None and in_eval:
            position = {
                "side": pending,
                "signal_time": enriched.iloc[i - 1]["timestamp"],
                "entry_time": ts,
                "entry_price": float(row["open"]),
                "entry_index": i,
            }
            pending = None
        elif not in_eval:
            pending = None

        if position is not None and in_eval:
            side = position["side"]
            exit_price = None
            reason = None
            if side == "LONG":
                band = row["executable_keltner_upper"]
                if pd.notna(band) and float(row["high"]) >= float(band):
                    exit_price = max(float(row["open"]), float(band))
                    reason = "KELTNER_UPPER_TOUCH"
            else:
                band = row["executable_keltner_lower"]
                if pd.notna(band) and float(row["low"]) <= float(band):
                    exit_price = min(float(row["open"]), float(band))
                    reason = "KELTNER_LOWER_TOUCH"

            if exit_price is not None:
                gross = ((exit_price / position["entry_price"] - 1.0) if side == "LONG" else (position["entry_price"] / exit_price - 1.0)) * 100.0
                trades.append({
                    **position,
                    "exit_time": ts,
                    "exit_price": exit_price,
                    "exit_reason": reason,
                    "bars_held": i - position["entry_index"] + 1,
                    "gross_return_pct": gross,
                    "cost_pct": cost,
                    "net_return_pct": gross - cost,
                })
                position = None

        # Signal at close t -> order for open t+1. Do not arm the final eval bar.
        if in_eval and position is None and pending is None and i + 1 < len(enriched):
            next_ts = pd.Timestamp(enriched.iloc[i + 1]["timestamp"])
            if next_ts < end:
                if bool(row["long_signal"]):
                    pending = "LONG"
                elif bool(row["short_signal"]):
                    pending = "SHORT"

    if position is not None:
        eligible = enriched[(enriched["timestamp"] >= start) & (enriched["timestamp"] < end)]
        last = eligible.iloc[-1]
        exit_price = float(last["close"])
        gross = ((exit_price / position["entry_price"] - 1.0) if position["side"] == "LONG" else (position["entry_price"] / exit_price - 1.0)) * 100.0
        trades.append({
            **position,
            "exit_time": last["timestamp"],
            "exit_price": exit_price,
            "exit_reason": "EVAL_END_MTM",
            "bars_held": int(eligible.index[-1]) - position["entry_index"] + 1,
            "gross_return_pct": gross,
            "cost_pct": cost,
            "net_return_pct": gross - cost,
        })

    return pd.DataFrame(trades)


def side_metrics(trades: pd.DataFrame, side: str) -> dict:
    if trades.empty:
        return metrics(trades)
    return metrics(trades[trades["side"] == side].reset_index(drop=True))


def main() -> None:
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    eval_start = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_START_UTC", "2026-06-17T00:00:00+00:00").replace("Z", "+00:00"))
    eval_end = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_END_UTC", "2026-09-15T00:00:00+00:00").replace("Z", "+00:00"))
    if eval_start.tzinfo is None:
        eval_start = eval_start.replace(tzinfo=timezone.utc)
    if eval_end.tzinfo is None:
        eval_end = eval_end.replace(tzinfo=timezone.utc)

    cfg = IchimokuKeltner1HConfig(
        fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")),
        slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")),
    )
    # 7 days is > 78 hours required for Senkou B 52 + 26 displacement.
    fetch_start = eval_start - timedelta(days=7)
    loader = BinanceDataVisionLoader()
    h1 = loader.fetch_window(symbol, "1h", fetch_start, eval_end)
    trades = run_window(h1, eval_start, eval_end, cfg)

    overall = metrics(trades)
    by_side = {"LONG": side_metrics(trades, "LONG"), "SHORT": side_metrics(trades, "SHORT")}
    monthly = {}
    if not trades.empty:
        month_key = pd.to_datetime(trades["entry_time"]).dt.strftime("%Y-%m")
        for month, group in trades.groupby(month_key):
            monthly[month] = metrics(group.reset_index(drop=True))

    out = Path("artifacts/ichimoku-keltner-1h")
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    manifest = {
        "benchmark": "ichimoku-keltner-1h-v1",
        "symbol": symbol,
        "timeframe": "1h",
        "evaluation_start_utc": eval_start.isoformat(),
        "evaluation_end_utc": eval_end.isoformat(),
        "entry": {
            "long": "completed 1h close crosses above plotted Leading Span A; enter next 1h open",
            "short": "completed 1h close crosses below plotted Leading Span B; enter next 1h open",
        },
        "ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52, "displacement": 26},
        "keltner": {"ema": 20, "atr": 20, "multiplier": 3.0, "atr_smoothing": "Wilder RMA"},
        "exit": {
            "long": "touch of previously-known Keltner upper 3.0",
            "short": "touch of previously-known Keltner lower 3.0",
            "protective_stop": None,
            "opposite_signal_exit": False,
            "evaluation_boundary": "open trade marked to market only for accounting",
        },
        "execution": "signal on close t; entry open t+1; Keltner touch threshold shifted one bar to prevent intrabar lookahead",
        "fee_bps_per_side": cfg.fee_bps_per_side,
        "slippage_bps_per_side": cfg.slippage_bps_per_side,
        "metrics": overall,
        "by_side": by_side,
        "by_entry_month": monthly,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    if not trades.empty:
        print("\nTRADES\n" + trades.to_string(index=False))


if __name__ == "__main__":
    main()
