from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators, metrics
from run_ichimoku_keltner_1h_backtest import run_window as run_control_window


STOP_ATR_MULTIPLIER = 2.0


def run_stop_window(
    data: pd.DataFrame,
    eval_start: datetime,
    eval_end: datetime,
    cfg: IchimokuKeltner1HConfig,
    stop_atr_multiplier: float = STOP_ATR_MULTIPLIER,
) -> pd.DataFrame:
    """Causal 1h benchmark with a fixed volatility-adjusted protective stop.

    The stop is fixed at entry using ATR20 from the last fully completed 1h bar:
      LONG  stop = entry - 2.0 * ATR20
      SHORT stop = entry + 2.0 * ATR20

    This is deliberately pre-registered as a single 2 ATR hypothesis rather
    than optimized on the known August loss. If target and stop are both
    touched inside the same OHLC candle, STOP wins (conservative ordering).
    Gaps through the stop fill at the bar open, never at the stale stop price.
    """
    enriched = enrich_indicators(data, cfg)
    start = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    trades: list[dict] = []
    position: dict | None = None
    pending: str | None = None
    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 100.0

    for i, row in enriched.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        in_eval = start <= ts < end

        if position is None and pending is not None and in_eval:
            prior_atr = enriched.iloc[i - 1]["keltner_atr"] if i > 0 else float("nan")
            if pd.isna(prior_atr):
                pending = None
            else:
                entry_price = float(row["open"])
                atr = float(prior_atr)
                stop_price = (
                    entry_price - stop_atr_multiplier * atr
                    if pending == "LONG"
                    else entry_price + stop_atr_multiplier * atr
                )
                position = {
                    "side": pending,
                    "signal_time": enriched.iloc[i - 1]["timestamp"],
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "entry_atr": atr,
                    "stop_price": stop_price,
                    "entry_index": i,
                }
                pending = None
        elif not in_eval:
            pending = None

        if position is not None and in_eval:
            side = position["side"]
            stop = float(position["stop_price"])
            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])

            if side == "LONG":
                target = row["executable_keltner_upper"]
                stop_hit = bar_low <= stop
                target_hit = pd.notna(target) and bar_high >= float(target)
                if stop_hit:
                    exit_price = min(bar_open, stop)
                    reason = "ATR2_PROTECTIVE_STOP"
                elif target_hit:
                    exit_price = max(bar_open, float(target))
                    reason = "KELTNER_UPPER_TOUCH"
                else:
                    exit_price = None
                    reason = None
            else:
                target = row["executable_keltner_lower"]
                stop_hit = bar_high >= stop
                target_hit = pd.notna(target) and bar_low <= float(target)
                if stop_hit:
                    exit_price = max(bar_open, stop)
                    reason = "ATR2_PROTECTIVE_STOP"
                elif target_hit:
                    exit_price = min(bar_open, float(target))
                    reason = "KELTNER_LOWER_TOUCH"
                else:
                    exit_price = None
                    reason = None

            if exit_price is not None:
                gross = (
                    (exit_price / position["entry_price"] - 1.0)
                    if side == "LONG"
                    else (position["entry_price"] / exit_price - 1.0)
                ) * 100.0
                trades.append(
                    {
                        **position,
                        "exit_time": ts,
                        "exit_price": exit_price,
                        "exit_reason": reason,
                        "bars_held": i - position["entry_index"] + 1,
                        "gross_return_pct": gross,
                        "cost_pct": cost,
                        "net_return_pct": gross - cost,
                    }
                )
                position = None

        # Completed close t -> entry at open t+1. Never arm final eval bar.
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
        gross = (
            (exit_price / position["entry_price"] - 1.0)
            if position["side"] == "LONG"
            else (position["entry_price"] / exit_price - 1.0)
        ) * 100.0
        trades.append(
            {
                **position,
                "exit_time": last["timestamp"],
                "exit_price": exit_price,
                "exit_reason": "EVAL_END_MTM",
                "bars_held": int(eligible.index[-1]) - position["entry_index"] + 1,
                "gross_return_pct": gross,
                "cost_pct": cost,
                "net_return_pct": gross - cost,
            }
        )

    return pd.DataFrame(trades)


def main() -> None:
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    eval_start = datetime.fromisoformat(
        os.getenv("BACKTEST_EVAL_START_UTC", "2026-08-01T00:00:00+00:00").replace("Z", "+00:00")
    )
    eval_end = datetime.fromisoformat(
        os.getenv("BACKTEST_EVAL_END_UTC", "2026-09-01T00:00:00+00:00").replace("Z", "+00:00")
    )
    if eval_start.tzinfo is None:
        eval_start = eval_start.replace(tzinfo=timezone.utc)
    if eval_end.tzinfo is None:
        eval_end = eval_end.replace(tzinfo=timezone.utc)

    cfg = IchimokuKeltner1HConfig(
        fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")),
        slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")),
    )

    loader = BinanceDataVisionLoader()
    h1 = loader.fetch_window(symbol, "1h", eval_start - timedelta(days=7), eval_end)
    control = run_control_window(h1, eval_start, eval_end, cfg)
    stopped = run_stop_window(h1, eval_start, eval_end, cfg)

    control_metrics = metrics(control)
    stop_metrics = metrics(stopped)

    # Regression guard: the stop experiment is valid only if it reproduces the
    # already-observed August control exactly before adding the new risk rule.
    if int(control_metrics["trades"]) != 11:
        raise AssertionError(f"Control regression failed: expected 11 trades, got {control_metrics['trades']}")
    expected_control_net = -9.405390839095036
    if abs(float(control_metrics["net_return_pct"]) - expected_control_net) > 1e-9:
        raise AssertionError(
            "Control regression failed: expected net "
            f"{expected_control_net}, got {control_metrics['net_return_pct']}"
        )

    out = Path("artifacts/ichimoku-keltner-1h-stop")
    out.mkdir(parents=True, exist_ok=True)
    control.to_csv(out / "control_trades.csv", index=False)
    stopped.to_csv(out / "atr2_stop_trades.csv", index=False)

    stop_counts = stopped["exit_reason"].value_counts().to_dict() if not stopped.empty else {}
    manifest = {
        "benchmark": "ichimoku-keltner-1h-protective-stop-v1",
        "symbol": symbol,
        "timeframe": "1h",
        "evaluation_start_utc": eval_start.isoformat(),
        "evaluation_end_utc": eval_end.isoformat(),
        "hypothesis": "A fixed 2 ATR20 stop caps violent adverse moves without retuning Ichimoku/Keltner entries or targets.",
        "pre_registered_stop": {
            "atr_period": 20,
            "atr_smoothing": "Wilder RMA",
            "multiplier": STOP_ATR_MULTIPLIER,
            "atr_information_time": "last fully completed 1h candle before entry",
            "long": "entry - 2.0 ATR20",
            "short": "entry + 2.0 ATR20",
            "same_bar_target_and_stop": "STOP wins conservatively",
            "gap_rule": "fill at bar open when open gaps beyond stop",
            "optimization": "none; one threshold tested",
        },
        "unchanged": {
            "entry": "Ichimoku plotted-cloud crossing on completed 1h close; enter next 1h open",
            "target": "previously-known Keltner EMA20/ATR20 x3 band",
            "fee_bps_per_side": cfg.fee_bps_per_side,
            "slippage_bps_per_side": cfg.slippage_bps_per_side,
        },
        "control": control_metrics,
        "atr2_stop": stop_metrics,
        "atr2_exit_reason_counts": stop_counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    print("\nATR2 STOP TRADES\n" + (stopped.to_string(index=False) if not stopped.empty else "NO TRADES"))


if __name__ == "__main__":
    main()
