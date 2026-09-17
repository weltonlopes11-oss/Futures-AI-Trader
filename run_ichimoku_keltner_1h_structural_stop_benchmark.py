from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators, metrics
from run_ichimoku_keltner_1h_stop_backtest import STOP_ATR_MULTIPLIER, run_stop_window

PIVOT_SPAN = 3
CONTROL_EXPECTED_TRADES = 22
CONTROL_EXPECTED_NET_RETURN_PCT = 2.081194239


def add_causal_swings(data: pd.DataFrame, span: int = PIVOT_SPAN) -> pd.DataFrame:
    """Add last confirmed swing levels without retroactive knowledge.

    A pivot at j is confirmed only at j+span after span right-hand bars have
    completed. The confirmed level becomes usable from the following bar.
    """
    out = data.copy().reset_index(drop=True)
    last_high = float("nan")
    last_low = float("nan")
    highs, lows = [], []
    for i in range(len(out)):
        j = i - span
        if j >= span:
            window = out.iloc[j - span : j + span + 1]
            center = out.iloc[j]
            if float(center["high"]) == float(window["high"].max()):
                last_high = float(center["high"])
            if float(center["low"]) == float(window["low"].min()):
                last_low = float(center["low"])
        highs.append(last_high)
        lows.append(last_low)
    out["last_confirmed_swing_high"] = pd.Series(highs).shift(1)
    out["last_confirmed_swing_low"] = pd.Series(lows).shift(1)
    return out


def run_variant(data, eval_start, eval_end, cfg, mode: str) -> pd.DataFrame:
    enriched = add_causal_swings(enrich_indicators(data, cfg))
    start = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 100.0
    trades = []
    position = None
    pending_entry = None
    pending_structural_exit = False

    for i, row in enriched.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        in_eval = start <= ts < end

        if position is not None and pending_structural_exit and in_eval:
            exit_price = float(row["open"])
            side = position["side"]
            gross = ((exit_price / position["entry_price"] - 1.0) if side == "LONG" else (position["entry_price"] / exit_price - 1.0)) * 100.0
            trades.append({**position, "exit_time": ts, "exit_price": exit_price, "exit_reason": f"{mode.upper()}_INVALIDATION", "bars_held": i - position["entry_index"] + 1, "gross_return_pct": gross, "cost_pct": cost, "net_return_pct": gross - cost})
            position = None
            pending_structural_exit = False

        if position is None and pending_entry is not None and in_eval:
            prior_atr = enriched.iloc[i - 1]["keltner_atr"] if i > 0 else float("nan")
            if pd.notna(prior_atr):
                entry = float(row["open"])
                atr = float(prior_atr)
                stop = entry - STOP_ATR_MULTIPLIER * atr if pending_entry == "LONG" else entry + STOP_ATR_MULTIPLIER * atr
                position = {"side": pending_entry, "signal_time": enriched.iloc[i - 1]["timestamp"], "entry_time": ts, "entry_price": entry, "entry_atr": atr, "stop_price": stop, "entry_index": i}
            pending_entry = None
        elif not in_eval:
            pending_entry = None

        if position is not None and in_eval:
            side = position["side"]
            stop = float(position["stop_price"])
            o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
            if side == "LONG":
                target = row["executable_keltner_upper"]
                stop_hit = l <= stop
                target_hit = pd.notna(target) and h >= float(target)
                if stop_hit:
                    exit_price, reason = min(o, stop), "ATR2_PROTECTIVE_STOP"
                elif target_hit:
                    exit_price, reason = max(o, float(target)), "KELTNER_UPPER_TOUCH"
                else:
                    exit_price = reason = None
            else:
                target = row["executable_keltner_lower"]
                stop_hit = h >= stop
                target_hit = pd.notna(target) and l <= float(target)
                if stop_hit:
                    exit_price, reason = max(o, stop), "ATR2_PROTECTIVE_STOP"
                elif target_hit:
                    exit_price, reason = min(o, float(target)), "KELTNER_LOWER_TOUCH"
                else:
                    exit_price = reason = None

            if exit_price is not None:
                gross = ((exit_price / position["entry_price"] - 1.0) if side == "LONG" else (position["entry_price"] / exit_price - 1.0)) * 100.0
                trades.append({**position, "exit_time": ts, "exit_price": exit_price, "exit_reason": reason, "bars_held": i - position["entry_index"] + 1, "gross_return_pct": gross, "cost_pct": cost, "net_return_pct": gross - cost})
                position = None
                pending_structural_exit = False

        if position is not None and in_eval and i + 1 < len(enriched):
            next_ts = pd.Timestamp(enriched.iloc[i + 1]["timestamp"])
            if next_ts < end:
                if mode == "kijun":
                    level = row["kijun"]
                    invalid = pd.notna(level) and ((position["side"] == "LONG" and float(row["close"]) < float(level)) or (position["side"] == "SHORT" and float(row["close"]) > float(level)))
                elif mode == "swing":
                    level = row["last_confirmed_swing_low"] if position["side"] == "LONG" else row["last_confirmed_swing_high"]
                    invalid = pd.notna(level) and ((position["side"] == "LONG" and float(row["close"]) < float(level)) or (position["side"] == "SHORT" and float(row["close"]) > float(level)))
                else:
                    raise ValueError(mode)
                pending_structural_exit = bool(invalid)

        if in_eval and position is None and pending_entry is None and i + 1 < len(enriched):
            next_ts = pd.Timestamp(enriched.iloc[i + 1]["timestamp"])
            if next_ts < end:
                if bool(row["long_signal"]):
                    pending_entry = "LONG"
                elif bool(row["short_signal"]):
                    pending_entry = "SHORT"

    if position is not None:
        eligible = enriched[(enriched["timestamp"] >= start) & (enriched["timestamp"] < end)]
        last = eligible.iloc[-1]
        exit_price = float(last["close"])
        side = position["side"]
        gross = ((exit_price / position["entry_price"] - 1.0) if side == "LONG" else (position["entry_price"] / exit_price - 1.0)) * 100.0
        trades.append({**position, "exit_time": last["timestamp"], "exit_price": exit_price, "exit_reason": "EVAL_END_MTM", "bars_held": int(eligible.index[-1]) - position["entry_index"] + 1, "gross_return_pct": gross, "cost_pct": cost, "net_return_pct": gross - cost})
    return pd.DataFrame(trades)


def main():
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    eval_start = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_START_UTC", "2026-08-01T00:00:00+00:00").replace("Z", "+00:00"))
    eval_end = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_END_UTC", "2026-09-01T00:00:00+00:00").replace("Z", "+00:00"))
    if eval_start.tzinfo is None: eval_start = eval_start.replace(tzinfo=timezone.utc)
    if eval_end.tzinfo is None: eval_end = eval_end.replace(tzinfo=timezone.utc)
    cfg = IchimokuKeltner1HConfig(fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")), slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")))
    data = BinanceDataVisionLoader().fetch_window(symbol, "1h", eval_start - timedelta(days=7), eval_end)

    control = run_stop_window(data, eval_start, eval_end, cfg)
    kijun = run_variant(data, eval_start, eval_end, cfg, "kijun")
    swing = run_variant(data, eval_start, eval_end, cfg, "swing")
    cm = metrics(control)
    if int(cm["trades"]) != CONTROL_EXPECTED_TRADES or abs(float(cm["net_return_pct"]) - CONTROL_EXPECTED_NET_RETURN_PCT) > 1e-9:
        raise AssertionError(f"2ATR control regression failed: {cm}")

    result = {
        "benchmark": "ichimoku-keltner-1h-structural-invalidation-v1",
        "symbol": symbol, "timeframe": "1h",
        "evaluation_start_utc": eval_start.isoformat(), "evaluation_end_utc": eval_end.isoformat(),
        "pre_registered_rules": {
            "control": "fixed 2ATR20 emergency stop",
            "kijun_plus_2atr": "completed close through Kijun invalidates; exit next open; 2ATR remains emergency stop",
            "swing_plus_2atr": f"completed close through last causally confirmed {PIVOT_SPAN}-left/{PIVOT_SPAN}-right swing invalidates; exit next open; 2ATR remains emergency stop",
            "priority": "intrabar 2ATR stop wins over target; target then structural close invalidation",
            "optimization": "none; no ATR threshold retuning",
        },
        "control_2atr": cm,
        "kijun_plus_2atr": metrics(kijun),
        "swing_plus_2atr": metrics(swing),
        "exit_counts": {
            "control_2atr": control["exit_reason"].value_counts().to_dict(),
            "kijun_plus_2atr": kijun["exit_reason"].value_counts().to_dict(),
            "swing_plus_2atr": swing["exit_reason"].value_counts().to_dict(),
        },
    }
    out = Path("artifacts/ichimoku-keltner-1h-structural-stop")
    out.mkdir(parents=True, exist_ok=True)
    control.to_csv(out / "control_2atr.csv", index=False)
    kijun.to_csv(out / "kijun_plus_2atr.csv", index=False)
    swing.to_csv(out / "swing_plus_2atr.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
