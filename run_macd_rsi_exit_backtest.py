from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators, metrics
from run_ichimoku_keltner_1h_stop_backtest import run_stop_window
from run_macd_entry_filter_backtest import (
    INITIAL_CAPITAL_BRL,
    LEVERAGE,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    leveraged_path,
    run_macd_filtered_window,
)

RSI_PERIOD = 14
RSI_LONG_ARM = 70.0
RSI_SHORT_ARM = 20.0


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    return rsi


def run_macd_rsi_exit_window(
    data: pd.DataFrame,
    eval_start: datetime,
    eval_end: datetime,
    cfg: IchimokuKeltner1HConfig,
) -> tuple[pd.DataFrame, dict]:
    enriched = enrich_indicators(data, cfg)

    fast = enriched["close"].ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    slow = enriched["close"].ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    enriched["macd_line"] = fast - slow
    enriched["macd_signal_line"] = enriched["macd_line"].ewm(
        span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL
    ).mean()
    enriched["rsi14"] = wilder_rsi(enriched["close"], RSI_PERIOD)

    start = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)

    trades: list[dict] = []
    position: dict | None = None
    pending_entry: str | None = None
    pending_rsi_exit = False
    accepted = {"long": 0, "short": 0}
    blocked = {"long": 0, "short": 0}
    rsi_stats = {
        "long_armed": 0,
        "short_armed": 0,
        "long_exits": 0,
        "short_exits": 0,
    }

    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 100.0

    for i, row in enriched.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        in_eval = start <= ts < end

        # RSI reversal is only known after the previous candle closes,
        # therefore execution occurs causally at the current candle open.
        if position is not None and pending_rsi_exit and in_eval:
            side = position["side"]
            exit_price = float(row["open"])
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
                    "exit_reason": "RSI70_REVERSAL_EXIT" if side == "LONG" else "RSI20_REVERSAL_EXIT",
                    "bars_held": i - position["entry_index"],
                    "gross_return_pct": gross,
                    "cost_pct": cost,
                    "net_return_pct": gross - cost,
                    "exit_rsi_signal": float(enriched.iloc[i - 1]["rsi14"]),
                }
            )
            rsi_stats["long_exits" if side == "LONG" else "short_exits"] += 1
            position = None
            pending_rsi_exit = False

        if position is None and pending_entry is not None and in_eval:
            prior_atr = enriched.iloc[i - 1]["keltner_atr"] if i > 0 else float("nan")
            if pd.isna(prior_atr):
                pending_entry = None
            else:
                entry_price = float(row["open"])
                atr = float(prior_atr)
                signal_row = enriched.iloc[i - 1]
                position = {
                    "side": pending_entry,
                    "signal_time": signal_row["timestamp"],
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "entry_atr": atr,
                    "stop_price": (
                        entry_price - 2.0 * atr
                        if pending_entry == "LONG"
                        else entry_price + 2.0 * atr
                    ),
                    "entry_index": i,
                    "signal_macd": float(signal_row["macd_line"]),
                    "signal_macd_signal": float(signal_row["macd_signal_line"]),
                    "entry_rsi": float(signal_row["rsi14"]) if pd.notna(signal_row["rsi14"]) else None,
                    "rsi_extreme_armed": False,
                    "rsi_extreme_time": pd.NaT,
                    "rsi_extreme_value": None,
                }
                pending_entry = None
        elif not in_eval:
            pending_entry = None
            pending_rsi_exit = False

        # Normal target/stop keep priority because they can occur intrabar
        # before an RSI close-based reversal becomes known.
        if position is not None and in_eval:
            side = position["side"]
            stop = float(position["stop_price"])
            o, h, l = float(row["open"]), float(row["high"]), float(row["low"])

            if side == "LONG":
                target = row["executable_keltner_upper"]
                if l <= stop:
                    exit_price, reason = min(o, stop), "ATR2_PROTECTIVE_STOP"
                elif pd.notna(target) and h >= float(target):
                    exit_price, reason = max(o, float(target)), "KELTNER_UPPER_TOUCH"
                else:
                    exit_price = reason = None
            else:
                target = row["executable_keltner_lower"]
                if h >= stop:
                    exit_price, reason = max(o, stop), "ATR2_PROTECTIVE_STOP"
                elif pd.notna(target) and l <= float(target):
                    exit_price, reason = min(o, float(target)), "KELTNER_LOWER_TOUCH"
                else:
                    exit_price = reason = None

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
                        "exit_rsi_signal": None,
                    }
                )
                position = None
                pending_rsi_exit = False

        # Arm/trigger RSI only after this completed candle and only while trade survives.
        if position is not None and in_eval and pd.notna(row["rsi14"]):
            rsi = float(row["rsi14"])
            side = position["side"]

            if side == "LONG":
                if not position["rsi_extreme_armed"] and rsi > RSI_LONG_ARM:
                    position["rsi_extreme_armed"] = True
                    position["rsi_extreme_time"] = ts
                    position["rsi_extreme_value"] = rsi
                    rsi_stats["long_armed"] += 1
                elif position["rsi_extreme_armed"] and rsi < RSI_LONG_ARM and i + 1 < len(enriched):
                    next_ts = pd.Timestamp(enriched.iloc[i + 1]["timestamp"])
                    if next_ts < end:
                        pending_rsi_exit = True
            else:
                if not position["rsi_extreme_armed"] and rsi < RSI_SHORT_ARM:
                    position["rsi_extreme_armed"] = True
                    position["rsi_extreme_time"] = ts
                    position["rsi_extreme_value"] = rsi
                    rsi_stats["short_armed"] += 1
                elif position["rsi_extreme_armed"] and rsi > RSI_SHORT_ARM and i + 1 < len(enriched):
                    next_ts = pd.Timestamp(enriched.iloc[i + 1]["timestamp"])
                    if next_ts < end:
                        pending_rsi_exit = True

        # Base entries remain MACD-filtered exactly as the prior experiment.
        if in_eval and position is None and pending_entry is None and not pending_rsi_exit and i + 1 < len(enriched):
            next_ts = pd.Timestamp(enriched.iloc[i + 1]["timestamp"])
            if next_ts < end:
                macd = row["macd_line"]
                signal = row["macd_signal_line"]

                if bool(row["long_signal"]):
                    if pd.notna(macd) and pd.notna(signal) and float(macd) > float(signal):
                        pending_entry = "LONG"
                        accepted["long"] += 1
                    else:
                        blocked["long"] += 1
                elif bool(row["short_signal"]):
                    if pd.notna(macd) and pd.notna(signal) and float(macd) < float(signal):
                        pending_entry = "SHORT"
                        accepted["short"] += 1
                    else:
                        blocked["short"] += 1

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
                "exit_rsi_signal": None,
            }
        )

    return pd.DataFrame(trades), {
        "macd_accepted": accepted,
        "macd_blocked": blocked,
        "rsi": rsi_stats,
    }


def main() -> None:
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    eval_start = datetime.fromisoformat(
        os.getenv("BACKTEST_EVAL_START_UTC", "2026-01-01T00:00:00+00:00").replace("Z", "+00:00")
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

    h1 = BinanceDataVisionLoader().fetch_window(
        symbol, "1h", eval_start - timedelta(days=7), eval_end
    )

    official = run_stop_window(h1, eval_start, eval_end, cfg)
    macd_only, macd_counts = run_macd_filtered_window(h1, eval_start, eval_end, cfg)
    macd_rsi, combined_counts = run_macd_rsi_exit_window(h1, eval_start, eval_end, cfg)

    result = {
        "experiment_id": "macd-entry-plus-rsi14-reversal-exit-jan-aug-2026",
        "status": "separate_experiment_not_promoted",
        "official_strategy_modified": False,
        "period": "2026-01-01 through 2026-08-31 UTC",
        "candidate_change_only_vs_macd": {
            "rsi_period": RSI_PERIOD,
            "long": "after RSI closes >70 while LONG is open, arm exit; after later close <70, exit next bar open",
            "short": "after RSI closes <20 while SHORT is open, arm exit; after later close >20, exit next bar open",
            "causality": "RSI decisions use completed 1h candles; exit executes next 1h open",
            "priority": "intrabar ATR stop and Keltner target are evaluated before the RSI close-based reversal",
        },
        "official": metrics(official),
        "macd_only": metrics(macd_only),
        "macd_plus_rsi_exit": metrics(macd_rsi),
        "official_leveraged_path": leveraged_path(official),
        "macd_only_leveraged_path": leveraged_path(macd_only),
        "macd_plus_rsi_leveraged_path": leveraged_path(macd_rsi),
        "macd_only_filter_counts": macd_counts,
        "combined_counts": combined_counts,
        "candidate_exit_reason_counts": (
            macd_rsi["exit_reason"].value_counts().to_dict() if not macd_rsi.empty else {}
        ),
        "unchanged": {
            "entry_base": "Ichimoku plotted-cloud crossing",
            "entry_filter": "MACD 12/26/9 direction filter from prior experiment",
            "target": "Keltner EMA20/ATR20 x3",
            "stop": "2 ATR20 using last completed candle ATR",
            "fees_bps_per_side": cfg.fee_bps_per_side,
            "slippage_bps_per_side": cfg.slippage_bps_per_side,
            "capital": {"initial_brl": INITIAL_CAPITAL_BRL, "leverage": LEVERAGE},
        },
    }

    out = Path("artifacts/macd-rsi-exit")
    out.mkdir(parents=True, exist_ok=True)
    official.to_csv(out / "official_trades.csv", index=False)
    macd_only.to_csv(out / "macd_only_trades.csv", index=False)
    macd_rsi.to_csv(out / "macd_rsi_trades.csv", index=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
