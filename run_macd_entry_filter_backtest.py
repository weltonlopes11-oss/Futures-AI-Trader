from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators, metrics
from run_ichimoku_keltner_1h_stop_backtest import run_stop_window

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
LEVERAGE = 10.0
INITIAL_CAPITAL_BRL = 2000.0


def add_macd(frame: pd.DataFrame) -> pd.DataFrame:
    out = enrich_indicators(frame)
    fast = out["close"].ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    slow = out["close"].ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    out["macd_line"] = fast - slow
    out["macd_signal_line"] = out["macd_line"].ewm(
        span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL
    ).mean()
    return out


def run_macd_filtered_window(
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

    start = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    trades, position, pending = [], None, None
    blocked = {"long": 0, "short": 0}
    accepted = {"long": 0, "short": 0}
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
                    entry_price - 2.0 * atr
                    if pending == "LONG"
                    else entry_price + 2.0 * atr
                )
                signal_row = enriched.iloc[i - 1]
                position = {
                    "side": pending,
                    "signal_time": signal_row["timestamp"],
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "entry_atr": atr,
                    "stop_price": stop_price,
                    "entry_index": i,
                    "signal_macd": float(signal_row["macd_line"]),
                    "signal_macd_signal": float(signal_row["macd_signal_line"]),
                }
                pending = None
        elif not in_eval:
            pending = None

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
                    }
                )
                position = None

        if in_eval and position is None and pending is None and i + 1 < len(enriched):
            next_ts = pd.Timestamp(enriched.iloc[i + 1]["timestamp"])
            if next_ts < end:
                macd = row["macd_line"]
                signal = row["macd_signal_line"]

                if bool(row["long_signal"]):
                    if pd.notna(macd) and pd.notna(signal) and float(macd) > float(signal):
                        pending = "LONG"
                        accepted["long"] += 1
                    else:
                        blocked["long"] += 1
                elif bool(row["short_signal"]):
                    if pd.notna(macd) and pd.notna(signal) and float(macd) < float(signal):
                        pending = "SHORT"
                        accepted["short"] += 1
                    else:
                        blocked["short"] += 1

    if position is not None:
        eligible = enriched[
            (enriched["timestamp"] >= start) & (enriched["timestamp"] < end)
        ]
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

    return pd.DataFrame(trades), {"accepted": accepted, "blocked": blocked}


def leveraged_path(trades: pd.DataFrame) -> dict:
    capital = INITIAL_CAPITAL_BRL
    peak = capital
    max_dd = 0.0
    rows = []

    if trades.empty:
        return {
            "initial_brl": INITIAL_CAPITAL_BRL,
            "final_brl": INITIAL_CAPITAL_BRL,
            "return_10x_pct": 0.0,
            "max_drawdown_10x_pct": 0.0,
            "by_exit_month": {},
        }

    ordered = trades.sort_values("exit_time").reset_index(drop=True)
    for _, trade in ordered.iterrows():
        start_capital = capital
        leveraged_return = LEVERAGE * float(trade["net_return_pct"]) / 100.0
        capital *= max(0.0, 1.0 + leveraged_return)
        peak = max(peak, capital)
        dd = 0.0 if peak <= 0 else (capital / peak - 1.0) * 100.0
        max_dd = min(max_dd, dd)
        rows.append(
            {
                "exit_month": pd.Timestamp(trade["exit_time"]).strftime("%Y-%m"),
                "start_brl": start_capital,
                "end_brl": capital,
                "simple_net_pct": float(trade["net_return_pct"]),
            }
        )

    path = pd.DataFrame(rows)
    by_month = {}
    for month, group in path.groupby("exit_month", sort=True):
        by_month[month] = {
            "trades": int(len(group)),
            "simple_net_pct": float(group["simple_net_pct"].sum()),
            "start_brl": float(group.iloc[0]["start_brl"]),
            "end_brl": float(group.iloc[-1]["end_brl"]),
        }

    return {
        "initial_brl": INITIAL_CAPITAL_BRL,
        "final_brl": float(capital),
        "return_10x_pct": float((capital / INITIAL_CAPITAL_BRL - 1.0) * 100.0),
        "max_drawdown_10x_pct": float(max_dd),
        "by_exit_month": by_month,
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

    warmup_start = eval_start - timedelta(days=7)
    h1 = BinanceDataVisionLoader().fetch_window(symbol, "1h", warmup_start, eval_end)

    official = run_stop_window(h1, eval_start, eval_end, cfg)
    candidate, filter_counts = run_macd_filtered_window(h1, eval_start, eval_end, cfg)

    official_metrics = metrics(official)
    candidate_metrics = metrics(candidate)
    official_path = leveraged_path(official)
    candidate_path = leveraged_path(candidate)

    result = {
        "experiment_id": "macd-12-26-9-entry-filter-jan-aug-2026",
        "status": "separate_experiment_not_promoted",
        "official_strategy_modified": False,
        "period": "2026-01-01 through 2026-08-31 UTC",
        "candidate_change_only": {
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "long_gate": "accept LONG signal only when MACD line > signal line on completed signal candle",
            "short_gate": "accept SHORT signal only when MACD line < signal line on completed signal candle",
            "execution": "accepted signal at close t -> enter at open t+1",
        },
        "unchanged": {
            "entry_base": "Ichimoku plotted-cloud crossing",
            "target": "Keltner EMA20/ATR20 x3",
            "stop": "2 ATR20 from entry using last completed candle ATR",
            "fees_bps_per_side": cfg.fee_bps_per_side,
            "slippage_bps_per_side": cfg.slippage_bps_per_side,
            "capital": {"initial_brl": INITIAL_CAPITAL_BRL, "leverage": LEVERAGE},
        },
        "filter_counts": filter_counts,
        "official": official_metrics,
        "candidate": candidate_metrics,
        "official_leveraged_path": official_path,
        "candidate_leveraged_path": candidate_path,
        "candidate_exit_reason_counts": (
            candidate["exit_reason"].value_counts().to_dict() if not candidate.empty else {}
        ),
    }

    out = Path("artifacts/macd-entry-filter")
    out.mkdir(parents=True, exist_ok=True)
    official.to_csv(out / "official_trades.csv", index=False)
    candidate.to_csv(out / "macd_filtered_trades.csv", index=False)
    (out / "result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
