from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators


def enrich_full_cloud_signals(
    frame: pd.DataFrame,
    cfg: IchimokuKeltner1HConfig | None = None,
) -> pd.DataFrame:
    """Experimental entry only: require a completed close outside the full Kumo."""
    cfg = cfg or IchimokuKeltner1HConfig()
    out = enrich_indicators(frame, cfg)

    cloud_top = out[["leading_span_a", "leading_span_b"]].max(axis=1)
    cloud_bottom = out[["leading_span_a", "leading_span_b"]].min(axis=1)
    prev_close = out["close"].shift(1)
    prev_top = cloud_top.shift(1)
    prev_bottom = cloud_bottom.shift(1)

    out["cloud_top"] = cloud_top
    out["cloud_bottom"] = cloud_bottom
    out["full_cloud_long_signal"] = (out["close"] > cloud_top) & (prev_close <= prev_top)
    out["full_cloud_short_signal"] = (out["close"] < cloud_bottom) & (prev_close >= prev_bottom)
    return out


def run_backtest_full_cloud_stop(
    frame: pd.DataFrame,
    start_utc: str,
    end_utc_exclusive: str,
    cfg: IchimokuKeltner1HConfig | None = None,
) -> pd.DataFrame:
    """Separate experiment; frozen live/forward strategy is not modified.

    Entry:
      LONG after a completed close crosses above max(plotted Span A, Span B).
      SHORT after a completed close crosses below min(plotted Span A, Span B).
      Execution is the next 1h candle open.

    Everything else stays aligned with the frozen 2ATR variant:
      Keltner EMA20/ATR20 x3 target, threshold shifted one bar;
      fixed 2ATR protective stop from signal-candle ATR;
      stop wins if stop and target are both touched in one bar;
      gaps beyond stop fill at bar open;
      4 bps fee + 1 bp slippage per side.
    """
    cfg = cfg or IchimokuKeltner1HConfig()
    data = enrich_full_cloud_signals(frame, cfg)
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc_exclusive)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")

    trades: list[dict] = []
    position: dict | None = None
    pending: dict | None = None
    round_trip_cost_pct = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 100.0

    for i, row in data.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts < start or ts >= end:
            continue

        if position is None and pending is not None:
            entry = float(row["open"])
            atr = float(pending["atr"])
            position = {
                "side": pending["side"],
                "signal_time": pending["signal_time"],
                "entry_time": row["timestamp"],
                "entry_price": entry,
                "entry_atr": atr,
                "stop_price": entry - 2.0 * atr if pending["side"] == "LONG" else entry + 2.0 * atr,
                "entry_index": i,
            }
            pending = None

        if position is not None:
            side = position["side"]
            stop = float(position["stop_price"])
            o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
            exit_price = None
            exit_reason = None

            if side == "LONG":
                target = row["executable_keltner_upper"]
                if l <= stop:
                    exit_price = min(o, stop)
                    exit_reason = "STOP_2ATR"
                elif pd.notna(target) and h >= float(target):
                    exit_price = max(o, float(target))
                    exit_reason = "KELTNER_UPPER_3ATR"
            else:
                target = row["executable_keltner_lower"]
                if h >= stop:
                    exit_price = max(o, stop)
                    exit_reason = "STOP_2ATR"
                elif pd.notna(target) and l <= float(target):
                    exit_price = min(o, float(target))
                    exit_reason = "KELTNER_LOWER_3ATR"

            if exit_price is not None:
                gross = (
                    (exit_price / position["entry_price"] - 1.0) * 100.0
                    if side == "LONG"
                    else (position["entry_price"] / exit_price - 1.0) * 100.0
                )
                trades.append(
                    {
                        **position,
                        "exit_time": row["timestamp"],
                        "exit_price": exit_price,
                        "exit_reason": exit_reason,
                        "bars_held": i - position["entry_index"] + 1,
                        "gross_return_pct": gross,
                        "cost_pct": round_trip_cost_pct,
                        "net_return_pct": gross - round_trip_cost_pct,
                    }
                )
                position = None

        if position is None and pending is None and i + 1 < len(data):
            next_ts = pd.Timestamp(data.iloc[i + 1]["timestamp"])
            if next_ts.tzinfo is None:
                next_ts = next_ts.tz_localize("UTC")
            if next_ts < end:
                if bool(row["full_cloud_long_signal"]):
                    pending = {"side": "LONG", "signal_time": row["timestamp"], "atr": float(row["keltner_atr"])}
                elif bool(row["full_cloud_short_signal"]):
                    pending = {"side": "SHORT", "signal_time": row["timestamp"], "atr": float(row["keltner_atr"])}

    if position is not None:
        eligible = data[(pd.to_datetime(data["timestamp"], utc=True) >= start) & (pd.to_datetime(data["timestamp"], utc=True) < end)]
        row = eligible.iloc[-1]
        exit_price = float(row["close"])
        gross = (
            (exit_price / position["entry_price"] - 1.0) * 100.0
            if position["side"] == "LONG"
            else (position["entry_price"] / exit_price - 1.0) * 100.0
        )
        trades.append(
            {
                **position,
                "exit_time": row["timestamp"],
                "exit_price": exit_price,
                "exit_reason": "EVAL_END_MTM",
                "bars_held": int(eligible.index[-1]) - position["entry_index"] + 1,
                "gross_return_pct": gross,
                "cost_pct": round_trip_cost_pct,
                "net_return_pct": gross - round_trip_cost_pct,
            }
        )

    return pd.DataFrame(trades)


def metrics_with_10x(trades: pd.DataFrame, initial_brl: float = 500.0, leverage: float = 10.0) -> dict:
    net = trades["net_return_pct"].astype(float) if not trades.empty else pd.Series(dtype=float)
    gains = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())

    equity = initial_brl
    peak = initial_brl
    min_equity = initial_brl
    max_equity = initial_brl
    max_dd = 0.0
    for ret in net:
        equity += equity * leverage * float(ret) / 100.0
        peak = max(peak, equity)
        min_equity = min(min_equity, equity)
        max_equity = max(max_equity, equity)
        max_dd = min(max_dd, (equity / peak - 1.0) * 100.0)

    additive = net.cumsum()
    additive_peak = additive.cummax().clip(lower=0.0) if len(additive) else additive
    return {
        "trades": int(len(trades)),
        "win_rate_pct": float((net > 0).mean() * 100.0) if len(net) else 0.0,
        "simple_net_pct": float(net.sum()),
        "compounded_pct": float((np.prod(1.0 + net / 100.0) - 1.0) * 100.0) if len(net) else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0),
        "expectancy_pct": float(net.mean()) if len(net) else 0.0,
        "max_drawdown_pct": float((additive - additive_peak).min()) if len(net) else 0.0,
        "long_trades": int((trades["side"] == "LONG").sum()) if len(trades) else 0,
        "short_trades": int((trades["side"] == "SHORT").sum()) if len(trades) else 0,
        "stops": int((trades["exit_reason"] == "STOP_2ATR").sum()) if len(trades) else 0,
        "targets": int(trades["exit_reason"].astype(str).str.startswith("KELTNER").sum()) if len(trades) else 0,
        "eval_end_mtm": int((trades["exit_reason"] == "EVAL_END_MTM").sum()) if len(trades) else 0,
        "avg_hours_held": float(trades["bars_held"].mean()) if len(trades) else 0.0,
        "median_hours_held": float(trades["bars_held"].median()) if len(trades) else 0.0,
        "leveraged_10x": {
            "initial_brl": initial_brl,
            "final_brl": equity,
            "return_pct": (equity / initial_brl - 1.0) * 100.0,
            "max_equity_brl": max_equity,
            "min_equity_brl": min_equity,
            "max_drawdown_pct": max_dd,
        },
    }
