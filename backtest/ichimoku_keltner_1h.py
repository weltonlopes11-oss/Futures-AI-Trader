from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IchimokuKeltner1HConfig:
    tenkan: int = 9
    kijun: int = 26
    senkou_b: int = 52
    displacement: int = 26
    keltner_ema: int = 20
    keltner_atr: int = 20
    keltner_multiplier: float = 3.0
    fee_bps_per_side: float = 4.0
    slippage_bps_per_side: float = 1.0


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def enrich_indicators(frame: pd.DataFrame, cfg: IchimokuKeltner1HConfig | None = None) -> pd.DataFrame:
    """Build standard Ichimoku 9/26/52/26 and Keltner EMA20/ATR20 x3.

    Senkou spans are shifted 26 bars forward exactly as plotted. Therefore the
    cloud value compared with price at time t was computed from information
    available 26 completed bars earlier and is causal.
    """
    cfg = cfg or IchimokuKeltner1HConfig()
    out = frame.copy().sort_values("timestamp").reset_index(drop=True)

    high9 = out["high"].rolling(cfg.tenkan, min_periods=cfg.tenkan).max()
    low9 = out["low"].rolling(cfg.tenkan, min_periods=cfg.tenkan).min()
    out["tenkan"] = (high9 + low9) / 2.0

    high26 = out["high"].rolling(cfg.kijun, min_periods=cfg.kijun).max()
    low26 = out["low"].rolling(cfg.kijun, min_periods=cfg.kijun).min()
    out["kijun"] = (high26 + low26) / 2.0

    out["leading_span_a"] = ((out["tenkan"] + out["kijun"]) / 2.0).shift(cfg.displacement)

    high52 = out["high"].rolling(cfg.senkou_b, min_periods=cfg.senkou_b).max()
    low52 = out["low"].rolling(cfg.senkou_b, min_periods=cfg.senkou_b).min()
    out["leading_span_b"] = ((high52 + low52) / 2.0).shift(cfg.displacement)

    out["keltner_mid"] = out["close"].ewm(span=cfg.keltner_ema, adjust=False, min_periods=cfg.keltner_ema).mean()
    tr = _true_range(out)
    out["keltner_atr"] = tr.ewm(alpha=1.0 / cfg.keltner_atr, adjust=False, min_periods=cfg.keltner_atr).mean()
    out["keltner_upper"] = out["keltner_mid"] + cfg.keltner_multiplier * out["keltner_atr"]
    out["keltner_lower"] = out["keltner_mid"] - cfg.keltner_multiplier * out["keltner_atr"]

    # A touch during bar t must use a threshold known before bar t opened.
    out["executable_keltner_upper"] = out["keltner_upper"].shift(1)
    out["executable_keltner_lower"] = out["keltner_lower"].shift(1)

    # Entry signal is a completed-candle crossing. Execution is next bar open.
    prev_close = out["close"].shift(1)
    prev_a = out["leading_span_a"].shift(1)
    prev_b = out["leading_span_b"].shift(1)
    out["long_signal"] = (out["close"] > out["leading_span_a"]) & (prev_close <= prev_a)
    out["short_signal"] = (out["close"] < out["leading_span_b"]) & (prev_close >= prev_b)
    return out


def run_backtest(frame: pd.DataFrame, cfg: IchimokuKeltner1HConfig | None = None) -> pd.DataFrame:
    """One-position-at-a-time causal backtest.

    LONG: close crosses above plotted Leading Span A; buy next bar open; close
    when a future bar high touches the previously-known Keltner upper x3.
    SHORT: symmetric below Leading Span B / Keltner lower x3.

    There is intentionally no protective stop. Opposite signals are ignored
    while a position is open. An open position at the evaluation boundary is
    marked to market at the final close with exit_reason=EVAL_END_MTM so the
    accounting does not hide unrealized PnL.
    """
    cfg = cfg or IchimokuKeltner1HConfig()
    data = enrich_indicators(frame, cfg)
    trades: list[dict] = []
    position: dict | None = None
    pending_side: str | None = None

    round_trip_cost_pct = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 100.0

    for i, row in data.iterrows():
        # Execute a signal generated on the previous completed candle.
        if position is None and pending_side is not None:
            position = {
                "side": pending_side,
                "entry_time": row["timestamp"],
                "entry_price": float(row["open"]),
                "signal_time": data.iloc[i - 1]["timestamp"] if i > 0 else pd.NaT,
                "entry_index": i,
            }
            pending_side = None

        if position is not None:
            side = position["side"]
            exit_price = None
            exit_reason = None
            if side == "LONG":
                band = row["executable_keltner_upper"]
                if pd.notna(band) and float(row["high"]) >= float(band):
                    # If the known target is already below the bar open, the
                    # executable fill cannot be worse than the opening price.
                    exit_price = max(float(row["open"]), float(band))
                    exit_reason = "KELTNER_UPPER_TOUCH"
            else:
                band = row["executable_keltner_lower"]
                if pd.notna(band) and float(row["low"]) <= float(band):
                    exit_price = min(float(row["open"]), float(band))
                    exit_reason = "KELTNER_LOWER_TOUCH"

            if exit_price is not None:
                if side == "LONG":
                    gross = (exit_price / position["entry_price"] - 1.0) * 100.0
                else:
                    gross = (position["entry_price"] / exit_price - 1.0) * 100.0
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

        # Signals are only armed after this candle closes, and only if flat.
        if position is None and pending_side is None and i + 1 < len(data):
            if bool(row["long_signal"]):
                pending_side = "LONG"
            elif bool(row["short_signal"]):
                pending_side = "SHORT"

    if position is not None and len(data):
        row = data.iloc[-1]
        exit_price = float(row["close"])
        if position["side"] == "LONG":
            gross = (exit_price / position["entry_price"] - 1.0) * 100.0
        else:
            gross = (position["entry_price"] / exit_price - 1.0) * 100.0
        trades.append(
            {
                **position,
                "exit_time": row["timestamp"],
                "exit_price": exit_price,
                "exit_reason": "EVAL_END_MTM",
                "bars_held": len(data) - position["entry_index"],
                "gross_return_pct": gross,
                "cost_pct": round_trip_cost_pct,
                "net_return_pct": gross - round_trip_cost_pct,
            }
        )

    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "net_return_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }
    net = trades["net_return_pct"].astype(float)
    gains = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())
    equity = net.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    return {
        "trades": int(len(trades)),
        "win_rate_pct": float((net > 0).mean() * 100.0),
        "net_return_pct": float(net.sum()),
        "compounded_return_pct": float((np.prod(1.0 + net / 100.0) - 1.0) * 100.0),
        "profit_factor": float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0),
        "expectancy_pct": float(net.mean()),
        "max_drawdown_pct": float(drawdown.min()) if len(drawdown) else 0.0,
        "avg_bars_held": float(trades["bars_held"].mean()),
        "median_bars_held": float(trades["bars_held"].median()),
        "long_trades": int((trades["side"] == "LONG").sum()),
        "short_trades": int((trades["side"] == "SHORT").sum()),
        "eval_end_mtm": int((trades["exit_reason"] == "EVAL_END_MTM").sum()),
    }
