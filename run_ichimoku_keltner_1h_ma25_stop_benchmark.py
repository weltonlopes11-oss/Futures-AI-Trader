from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators, metrics
from run_ichimoku_keltner_1h_stop_backtest import STOP_ATR_MULTIPLIER, run_stop_window

MA_PERIOD = 25
CONTROL_EXPECTED_TRADES = 22
CONTROL_EXPECTED_NET_RETURN_PCT = 2.0811942390375644


def run_ma25_variant(data, eval_start, eval_end, cfg):
    enriched = enrich_indicators(data, cfg).copy().reset_index(drop=True)
    # Simple moving average of the last 25 completed 1h closes. On bar t the
    # MA25 is evaluated only after that bar has closed; invalidation executes
    # at the next bar open, so there is no look-ahead.
    enriched["ma25"] = enriched["close"].rolling(MA_PERIOD, min_periods=MA_PERIOD).mean()

    start = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 100.0
    trades = []
    position = None
    pending_entry = None
    pending_ma_exit = False

    for i, row in enriched.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        in_eval = start <= ts < end

        if position is not None and pending_ma_exit and in_eval:
            exit_price = float(row["open"])
            side = position["side"]
            gross = ((exit_price / position["entry_price"] - 1.0) if side == "LONG" else (position["entry_price"] / exit_price - 1.0)) * 100.0
            trades.append({**position, "exit_time": ts, "exit_price": exit_price, "exit_reason": "MA25_INVALIDATION", "bars_held": i - position["entry_index"] + 1, "gross_return_pct": gross, "cost_pct": cost, "net_return_pct": gross - cost})
            position = None
            pending_ma_exit = False

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
                pending_ma_exit = False

        # MA25 is a thesis invalidation, evaluated only on the completed close
        # and executed at the following open. The 2ATR stop remains the hard
        # intrabar emergency stop and has priority over the Keltner target.
        if position is not None and in_eval and i + 1 < len(enriched):
            next_ts = pd.Timestamp(enriched.iloc[i + 1]["timestamp"])
            if next_ts < end and pd.notna(row["ma25"]):
                pending_ma_exit = bool(
                    (position["side"] == "LONG" and float(row["close"]) < float(row["ma25"]))
                    or (position["side"] == "SHORT" and float(row["close"]) > float(row["ma25"]))
                )

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
    ma25 = run_ma25_variant(data, eval_start, eval_end, cfg)
    cm = metrics(control)
    if int(cm["trades"]) != CONTROL_EXPECTED_TRADES or abs(float(cm["net_return_pct"]) - CONTROL_EXPECTED_NET_RETURN_PCT) > 1e-9:
        raise AssertionError(f"2ATR control regression failed: {cm}")

    result = {
        "benchmark": "ichimoku-keltner-1h-ma25-invalidation-v1",
        "symbol": symbol,
        "timeframe": "1h",
        "evaluation_start_utc": eval_start.isoformat(),
        "evaluation_end_utc": eval_end.isoformat(),
        "pre_registered_rules": {
            "ma_type": "SMA",
            "ma_period": MA_PERIOD,
            "long_invalidation": "completed 1h close below SMA25; exit next open",
            "short_invalidation": "completed 1h close above SMA25; exit next open",
            "emergency_stop": "fixed 2ATR20 from entry",
            "target": "unchanged causal Keltner 3.0 boundary",
            "priority": "intrabar 2ATR stop first, Keltner target second, MA25 close invalidation third",
            "optimization": "none; one preregistered MA25 test",
        },
        "control_2atr": cm,
        "ma25_plus_2atr": metrics(ma25),
        "exit_counts": {
            "control_2atr": control["exit_reason"].value_counts().to_dict(),
            "ma25_plus_2atr": ma25["exit_reason"].value_counts().to_dict(),
        },
    }
    out = Path("artifacts/ichimoku-keltner-1h-ma25-stop")
    out.mkdir(parents=True, exist_ok=True)
    control.to_csv(out / "control_2atr.csv", index=False)
    ma25.to_csv(out / "ma25_plus_2atr.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
