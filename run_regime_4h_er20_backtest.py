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
from run_regime_4h_ema50_backtest import (
    INITIAL_CAPITAL_BRL,
    LEVERAGE,
    attach_4h_regime,
    leveraged_path,
    run_regime_window,
)

ER_PERIOD = 20
ER_THRESHOLD = 0.25


def add_kaufman_er(frame: pd.DataFrame, period: int = ER_PERIOD) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].astype(float)
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period, min_periods=period).sum()
    out["er20"] = direction / volatility.replace(0.0, np.nan)
    return out


def run_regime_er_window(
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    eval_start: datetime,
    eval_end: datetime,
    cfg: IchimokuKeltner1HConfig,
) -> tuple[pd.DataFrame, dict]:
    enriched = add_kaufman_er(enrich_indicators(h1, cfg), ER_PERIOD)
    enriched = attach_4h_regime(enriched, h4)

    start = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)

    trades: list[dict] = []
    position: dict | None = None
    pending: dict | None = None

    counts = {
        "long": {"accepted": 0, "blocked_4h": 0, "blocked_er": 0},
        "short": {"accepted": 0, "blocked_4h": 0, "blocked_er": 0},
    }
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
                side = pending["side"]
                position = {
                    "side": side,
                    "signal_time": pending["signal_time"],
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "entry_atr": atr,
                    "stop_price": (
                        entry_price - 2.0 * atr if side == "LONG" else entry_price + 2.0 * atr
                    ),
                    "entry_index": i,
                    "regime_4h": pending["regime_4h"],
                    "regime_4h_close": pending["regime_4h_close"],
                    "ema50_4h": pending["ema50_4h"],
                    "ema50_slope": pending["ema50_slope"],
                    "er20": pending["er20"],
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
            if next_ts >= end:
                continue

            regime = row.get("regime_4h")
            if pd.isna(regime):
                regime = "NEUTRAL"
            er = row.get("er20")

            if bool(row["long_signal"]):
                if regime != "BULLISH":
                    counts["long"]["blocked_4h"] += 1
                elif pd.isna(er) or float(er) <= ER_THRESHOLD:
                    counts["long"]["blocked_er"] += 1
                else:
                    counts["long"]["accepted"] += 1
                    pending = {
                        "side": "LONG",
                        "signal_time": row["timestamp"],
                        "regime_4h": regime,
                        "regime_4h_close": float(row["regime_4h_close"]),
                        "ema50_4h": float(row["ema50_4h"]),
                        "ema50_slope": float(row["ema50_slope"]),
                        "er20": float(er),
                    }

            elif bool(row["short_signal"]):
                if regime != "BEARISH":
                    counts["short"]["blocked_4h"] += 1
                elif pd.isna(er) or float(er) <= ER_THRESHOLD:
                    counts["short"]["blocked_er"] += 1
                else:
                    counts["short"]["accepted"] += 1
                    pending = {
                        "side": "SHORT",
                        "signal_time": row["timestamp"],
                        "regime_4h": regime,
                        "regime_4h_close": float(row["regime_4h_close"]),
                        "ema50_4h": float(row["ema50_4h"]),
                        "ema50_slope": float(row["ema50_slope"]),
                        "er20": float(er),
                    }

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

    return pd.DataFrame(trades), counts


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

    loader = BinanceDataVisionLoader()
    warmup_start = eval_start - timedelta(days=30)
    h1 = loader.fetch_window(symbol, "1h", warmup_start, eval_end)
    h4 = loader.fetch_window(symbol, "4h", warmup_start - timedelta(days=15), eval_end)

    official = run_stop_window(h1, eval_start, eval_end, cfg)
    regime_only, regime_counts = run_regime_window(h1, h4, eval_start, eval_end, cfg)
    candidate, combined_counts = run_regime_er_window(h1, h4, eval_start, eval_end, cfg)

    result = {
        "experiment_id": "4h-ema50-plus-1h-kaufman-er20-025-jan-aug-2026",
        "status": "separate_experiment_not_promoted",
        "official_strategy_modified": False,
        "period": "2026-01-01 through 2026-08-31 UTC",
        "candidate_change_only_vs_regime_4h": {
            "er_timeframe": "1h completed signal candle",
            "period": ER_PERIOD,
            "threshold": ER_THRESHOLD,
            "gate": "ER20 > 0.25",
            "formula": "abs(close_t-close_t-20) / sum(abs(delta_close), 20)",
            "causality": "ER computed only from completed 1h candles; accepted signal enters next 1h open",
        },
        "unchanged": {
            "regime_4h": "Close vs EMA50 plus EMA50 slope from latest fully completed 4h candle",
            "entry": "official 1h Ichimoku signal",
            "target": "Keltner EMA20/ATR20 x3",
            "stop": "2 ATR20 using last completed 1h candle ATR",
            "fees_bps_per_side": cfg.fee_bps_per_side,
            "slippage_bps_per_side": cfg.slippage_bps_per_side,
            "capital": {"initial_brl": INITIAL_CAPITAL_BRL, "leverage": LEVERAGE},
        },
        "official": metrics(official),
        "regime_4h_only": metrics(regime_only),
        "regime_4h_plus_er20": metrics(candidate),
        "official_leveraged_path": leveraged_path(official),
        "regime_4h_only_leveraged_path": leveraged_path(regime_only),
        "regime_4h_plus_er20_leveraged_path": leveraged_path(candidate),
        "regime_4h_only_counts": regime_counts,
        "combined_gate_counts": combined_counts,
        "candidate_exit_reason_counts": (
            candidate["exit_reason"].value_counts().to_dict() if not candidate.empty else {}
        ),
    }

    out = Path("artifacts/regime-4h-er20")
    out.mkdir(parents=True, exist_ok=True)
    official.to_csv(out / "official_trades.csv", index=False)
    regime_only.to_csv(out / "regime_4h_only_trades.csv", index=False)
    candidate.to_csv(out / "regime_4h_er20_trades.csv", index=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
