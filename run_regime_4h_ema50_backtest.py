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

EMA_4H_PERIOD = 50
LEVERAGE = 10.0
INITIAL_CAPITAL_BRL = 2000.0


def build_4h_regime(h4: pd.DataFrame) -> pd.DataFrame:
    """Build a causal 4H EMA50 regime table.

    Binance timestamps are bar-open times. A 4H regime observation is only
    available after that 4H candle closes, so availability_time = open + 4h.
    """
    out = h4.copy().sort_values("timestamp").reset_index(drop=True)
    out["ema50_4h"] = out["close"].ewm(
        span=EMA_4H_PERIOD,
        adjust=False,
        min_periods=EMA_4H_PERIOD,
    ).mean()
    out["ema50_4h_prev"] = out["ema50_4h"].shift(1)
    out["ema50_slope"] = out["ema50_4h"] - out["ema50_4h_prev"]

    bullish = (out["close"] > out["ema50_4h"]) & (out["ema50_slope"] > 0)
    bearish = (out["close"] < out["ema50_4h"]) & (out["ema50_slope"] < 0)

    out["regime_4h"] = np.select(
        [bullish, bearish],
        ["BULLISH", "BEARISH"],
        default="NEUTRAL",
    )
    out["regime_4h_close"] = out["close"].astype(float)
    out["regime_available_time"] = pd.to_datetime(out["timestamp"]) + pd.Timedelta(hours=4)

    return out[
        [
            "regime_available_time",
            "regime_4h",
            "regime_4h_close",
            "ema50_4h",
            "ema50_slope",
        ]
    ].dropna(subset=["ema50_4h", "ema50_slope"])


def attach_4h_regime(h1_enriched: pd.DataFrame, h4: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest fully completed 4H regime to each completed 1H signal candle."""
    one = h1_enriched.copy().sort_values("timestamp").reset_index(drop=True)
    one["signal_available_time"] = pd.to_datetime(one["timestamp"]) + pd.Timedelta(hours=1)

    regime = build_4h_regime(h4).sort_values("regime_available_time").reset_index(drop=True)

    merged = pd.merge_asof(
        one.sort_values("signal_available_time"),
        regime,
        left_on="signal_available_time",
        right_on="regime_available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("timestamp").reset_index(drop=True)


def run_regime_window(
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    eval_start: datetime,
    eval_end: datetime,
    cfg: IchimokuKeltner1HConfig,
) -> tuple[pd.DataFrame, dict]:
    enriched = attach_4h_regime(enrich_indicators(h1, cfg), h4)

    start = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)

    trades: list[dict] = []
    position: dict | None = None
    pending: dict | None = None

    counts = {
        "long": {"accepted_bullish": 0, "blocked_bearish": 0, "blocked_neutral": 0},
        "short": {"accepted_bearish": 0, "blocked_bullish": 0, "blocked_neutral": 0},
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
                    "regime_available_time": pending["regime_available_time"],
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
                regime = row.get("regime_4h")
                if pd.isna(regime):
                    regime = "NEUTRAL"

                if bool(row["long_signal"]):
                    if regime == "BULLISH":
                        counts["long"]["accepted_bullish"] += 1
                        pending = {
                            "side": "LONG",
                            "signal_time": row["timestamp"],
                            "regime_4h": regime,
                            "regime_4h_close": float(row["regime_4h_close"]),
                            "ema50_4h": float(row["ema50_4h"]),
                            "ema50_slope": float(row["ema50_slope"]),
                            "regime_available_time": row["regime_available_time"],
                        }
                    elif regime == "BEARISH":
                        counts["long"]["blocked_bearish"] += 1
                    else:
                        counts["long"]["blocked_neutral"] += 1

                elif bool(row["short_signal"]):
                    if regime == "BEARISH":
                        counts["short"]["accepted_bearish"] += 1
                        pending = {
                            "side": "SHORT",
                            "signal_time": row["timestamp"],
                            "regime_4h": regime,
                            "regime_4h_close": float(row["regime_4h_close"]),
                            "ema50_4h": float(row["ema50_4h"]),
                            "ema50_slope": float(row["ema50_slope"]),
                            "regime_available_time": row["regime_available_time"],
                        }
                    elif regime == "BULLISH":
                        counts["short"]["blocked_bullish"] += 1
                    else:
                        counts["short"]["blocked_neutral"] += 1

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

    return pd.DataFrame(trades), counts


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

    loader = BinanceDataVisionLoader()
    warmup_start = eval_start - timedelta(days=30)
    h1 = loader.fetch_window(symbol, "1h", warmup_start, eval_end)
    h4 = loader.fetch_window(symbol, "4h", warmup_start - timedelta(days=15), eval_end)

    official = run_stop_window(h1, eval_start, eval_end, cfg)
    candidate, regime_counts = run_regime_window(h1, h4, eval_start, eval_end, cfg)

    result = {
        "experiment_id": "4h-ema50-directional-regime-gate-jan-aug-2026",
        "status": "separate_experiment_not_promoted",
        "official_strategy_modified": False,
        "period": "2026-01-01 through 2026-08-31 UTC",
        "candidate_change_only": {
            "timeframe": "4h",
            "ema_period": 50,
            "bullish": "latest fully completed 4h close > EMA50 and EMA50(current) > EMA50(previous)",
            "bearish": "latest fully completed 4h close < EMA50 and EMA50(current) < EMA50(previous)",
            "neutral": "all other cases; block new entry",
            "long_gate": "allow 1h LONG only in BULLISH regime",
            "short_gate": "allow 1h SHORT only in BEARISH regime",
            "causality": "1h signal uses only the latest fully completed 4h candle available at the 1h signal close",
        },
        "unchanged": {
            "entry": "official 1h Ichimoku signal",
            "target": "Keltner EMA20/ATR20 x3",
            "stop": "2 ATR20 using last completed 1h candle ATR",
            "fees_bps_per_side": cfg.fee_bps_per_side,
            "slippage_bps_per_side": cfg.slippage_bps_per_side,
            "capital": {"initial_brl": INITIAL_CAPITAL_BRL, "leverage": LEVERAGE},
        },
        "regime_gate_counts": regime_counts,
        "official": metrics(official),
        "candidate": metrics(candidate),
        "official_leveraged_path": leveraged_path(official),
        "candidate_leveraged_path": leveraged_path(candidate),
        "candidate_exit_reason_counts": (
            candidate["exit_reason"].value_counts().to_dict() if not candidate.empty else {}
        ),
    }

    out = Path("artifacts/regime-4h-ema50")
    out.mkdir(parents=True, exist_ok=True)
    official.to_csv(out / "official_trades.csv", index=False)
    candidate.to_csv(out / "regime_4h_trades.csv", index=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
