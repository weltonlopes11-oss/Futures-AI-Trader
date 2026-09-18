from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators, metrics
from run_regime_4h_ema50_backtest import (
    INITIAL_CAPITAL_BRL,
    LEVERAGE,
    attach_4h_regime,
    leveraged_path,
)

VOL_LOOKBACK_HOURS = 24 * 90
ER_PERIOD = 20
MIN_DISCOVERY_TRADES = 5
MIN_DISCOVERY_PF = 1.05


def add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("timestamp").reset_index(drop=True)

    close = out["close"].astype(float)
    atr = out["keltner_atr"].astype(float)
    out["atr_price"] = atr / close.replace(0.0, np.nan)

    minp = max(24 * 30, VOL_LOOKBACK_HOURS // 3)
    out["atr_price_q33"] = out["atr_price"].rolling(
        VOL_LOOKBACK_HOURS, min_periods=minp
    ).quantile(0.33)
    out["atr_price_q67"] = out["atr_price"].rolling(
        VOL_LOOKBACK_HOURS, min_periods=minp
    ).quantile(0.67)

    out["vol_regime"] = np.select(
        [
            out["atr_price"] <= out["atr_price_q33"],
            out["atr_price"] <= out["atr_price_q67"],
        ],
        ["LOW", "NORMAL"],
        default="HIGH",
    )
    out.loc[
        out[["atr_price", "atr_price_q33", "atr_price_q67"]].isna().any(axis=1),
        "vol_regime",
    ] = "UNKNOWN"

    direction = (close - close.shift(ER_PERIOD)).abs()
    path = close.diff().abs().rolling(ER_PERIOD, min_periods=ER_PERIOD).sum()
    out["er20"] = direction / path.replace(0.0, np.nan)

    out["efficiency_regime"] = np.select(
        [out["er20"] < 0.15, out["er20"] < 0.30],
        ["CHOP", "MIXED"],
        default="EFFICIENT",
    )
    out.loc[out["er20"].isna(), "efficiency_regime"] = "UNKNOWN"
    return out


def run_featured_regime_window(
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    eval_start: datetime,
    eval_end: datetime,
    cfg: IchimokuKeltner1HConfig,
    allowed_regimes: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    enriched = add_regime_features(enrich_indicators(h1, cfg))
    enriched = attach_4h_regime(enriched, h4)

    start = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)

    trades: list[dict] = []
    position: dict | None = None
    pending: dict | None = None

    counts = {
        "accepted": 0,
        "blocked_4h": 0,
        "blocked_regime": 0,
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
                    "stop_price": entry_price - 2.0 * atr if side == "LONG" else entry_price + 2.0 * atr,
                    "entry_index": i,
                    "regime_key": pending["regime_key"],
                    "vol_regime": pending["vol_regime"],
                    "efficiency_regime": pending["efficiency_regime"],
                    "atr_price": pending["atr_price"],
                    "er20": pending["er20"],
                    "regime_4h": pending["regime_4h"],
                    "ema50_4h": pending["ema50_4h"],
                    "ema50_slope": pending["ema50_slope"],
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

            regime_4h = row.get("regime_4h")
            if pd.isna(regime_4h):
                regime_4h = "NEUTRAL"

            if bool(row["long_signal"]):
                side = "LONG"
                required_4h = "BULLISH"
            elif bool(row["short_signal"]):
                side = "SHORT"
                required_4h = "BEARISH"
            else:
                continue

            if regime_4h != required_4h:
                counts["blocked_4h"] += 1
                continue

            vol_regime = str(row.get("vol_regime", "UNKNOWN"))
            efficiency_regime = str(row.get("efficiency_regime", "UNKNOWN"))
            regime_key = f"{side}|{vol_regime}|{efficiency_regime}"

            if allowed_regimes is not None and regime_key not in allowed_regimes:
                counts["blocked_regime"] += 1
                continue

            if vol_regime == "UNKNOWN" or efficiency_regime == "UNKNOWN":
                counts["blocked_regime"] += 1
                continue

            counts["accepted"] += 1
            pending = {
                "side": side,
                "signal_time": row["timestamp"],
                "regime_key": regime_key,
                "vol_regime": vol_regime,
                "efficiency_regime": efficiency_regime,
                "atr_price": float(row["atr_price"]),
                "er20": float(row["er20"]),
                "regime_4h": regime_4h,
                "ema50_4h": float(row["ema50_4h"]),
                "ema50_slope": float(row["ema50_slope"]),
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


def summarize_regimes(trades: pd.DataFrame) -> list[dict]:
    if trades.empty:
        return []

    rows = []
    for key, group in trades.groupby("regime_key", sort=True):
        vals = group["net_return_pct"].astype(float)
        wins = vals[vals > 0].sum()
        losses = -vals[vals < 0].sum()
        pf = float(wins / losses) if losses > 0 else float("inf")
        rows.append(
            {
                "regime_key": key,
                "trades": int(len(group)),
                "win_rate_pct": float((vals > 0).mean() * 100.0),
                "net_return_pct": float(vals.sum()),
                "expectancy_pct": float(vals.mean()),
                "profit_factor": pf,
                "max_loss_pct": float(vals.min()),
                "max_win_pct": float(vals.max()),
            }
        )
    return rows


def main() -> None:
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    cfg = IchimokuKeltner1HConfig(
        fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")),
        slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")),
    )

    discovery_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    discovery_end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    validation_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    validation_end = datetime(2026, 9, 1, tzinfo=timezone.utc)

    loader = BinanceDataVisionLoader()
    warmup_start = discovery_start - timedelta(days=120)
    h1 = loader.fetch_window(symbol, "1h", warmup_start, validation_end)
    h4 = loader.fetch_window(symbol, "4h", warmup_start - timedelta(days=15), validation_end)

    discovery_all, discovery_counts = run_featured_regime_window(
        h1, h4, discovery_start, discovery_end, cfg, allowed_regimes=None
    )
    discovery_summary = summarize_regimes(discovery_all)

    favorable = {
        row["regime_key"]
        for row in discovery_summary
        if row["trades"] >= MIN_DISCOVERY_TRADES
        and row["profit_factor"] >= MIN_DISCOVERY_PF
        and row["expectancy_pct"] > 0
    }

    discovery_favorable, _ = run_featured_regime_window(
        h1, h4, discovery_start, discovery_end, cfg, allowed_regimes=favorable
    )
    validation_all, validation_counts = run_featured_regime_window(
        h1, h4, validation_start, validation_end, cfg, allowed_regimes=None
    )
    validation_summary = summarize_regimes(validation_all)
    validation_favorable, validation_fav_counts = run_featured_regime_window(
        h1, h4, validation_start, validation_end, cfg, allowed_regimes=favorable
    )

    result = {
        "experiment_id": "regime-discovery-2025-validation-2026",
        "status": "research_only_not_promoted",
        "official_strategy_modified": False,
        "regime_definition": {
            "direction": "4H EMA50 gate: LONG only bullish; SHORT only bearish",
            "volatility": {
                "measure": "1H ATR20 / close",
                "adaptive_window": "trailing 90 days",
                "LOW": "<= trailing 33rd percentile",
                "NORMAL": "33rd-67th percentile",
                "HIGH": "> trailing 67th percentile",
            },
            "efficiency": {
                "measure": "Kaufman ER20 on completed 1H signal candle",
                "CHOP": "< 0.15",
                "MIXED": "0.15 to <0.30",
                "EFFICIENT": ">= 0.30",
            },
            "regime_key": "SIDE|VOLATILITY|EFFICIENCY",
        },
        "discovery_rule": {
            "period": "2025-01-01 through 2025-12-31 UTC",
            "min_trades": MIN_DISCOVERY_TRADES,
            "min_profit_factor": MIN_DISCOVERY_PF,
            "expectancy": "> 0",
            "favorable_regimes": sorted(favorable),
        },
        "discovery_all": {
            "metrics": metrics(discovery_all),
            "leveraged": leveraged_path(discovery_all),
            "gate_counts": discovery_counts,
            "by_regime": discovery_summary,
        },
        "discovery_favorable_only": {
            "metrics": metrics(discovery_favorable),
            "leveraged": leveraged_path(discovery_favorable),
        },
        "validation_all": {
            "period": "2026-01-01 through 2026-08-31 UTC",
            "metrics": metrics(validation_all),
            "leveraged": leveraged_path(validation_all),
            "gate_counts": validation_counts,
            "by_regime": validation_summary,
        },
        "validation_favorable_only": {
            "metrics": metrics(validation_favorable),
            "leveraged": leveraged_path(validation_favorable),
            "gate_counts": validation_fav_counts,
        },
    }

    out = Path("artifacts/regime-discovery")
    out.mkdir(parents=True, exist_ok=True)
    discovery_all.to_csv(out / "discovery_2025_all_regime_trades.csv", index=False)
    discovery_favorable.to_csv(out / "discovery_2025_favorable_only.csv", index=False)
    validation_all.to_csv(out / "validation_2026_all_regime_trades.csv", index=False)
    validation_favorable.to_csv(out / "validation_2026_favorable_only.csv", index=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
