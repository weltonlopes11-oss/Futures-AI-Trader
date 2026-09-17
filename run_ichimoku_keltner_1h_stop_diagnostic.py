from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, enrich_indicators
from run_ichimoku_keltner_1h_stop_backtest import run_stop_window


def _pct(side: str, entry: float, price: float) -> float:
    if side == "LONG":
        return (price / entry - 1.0) * 100.0
    return (entry / price - 1.0) * 100.0


def diagnose_stopped_trades(
    data: pd.DataFrame,
    stopped_trades: pd.DataFrame,
    eval_end: datetime,
    cfg: IchimokuKeltner1HConfig,
) -> pd.DataFrame:
    """Counterfactual path analysis for trades actually closed by the 2 ATR stop.

    This does NOT alter the strategy and does not re-enter positions. For each
    observed ATR2 stop, it asks what happened to that same entry afterward if
    the stop were ignored while the original moving Keltner target logic kept
    running. It reports whether that target was eventually touched, how long it
    took, and the full favorable/adverse excursion from entry through that
    counterfactual horizon.

    The purpose is diagnostic only: a later target touch does not mean the stop
    was wrong, because the path may first require an unacceptable drawdown.
    """
    enriched = enrich_indicators(data, cfg).reset_index(drop=True)
    end = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    cost = 2.0 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side) / 100.0
    rows: list[dict] = []

    stopped = stopped_trades[stopped_trades["exit_reason"] == "ATR2_PROTECTIVE_STOP"].copy()
    for _, trade in stopped.iterrows():
        side = str(trade["side"])
        entry = float(trade["entry_price"])
        entry_time = pd.Timestamp(trade["entry_time"])
        stop_time = pd.Timestamp(trade["exit_time"])

        path = enriched[(enriched["timestamp"] >= entry_time) & (enriched["timestamp"] < end)].copy()
        if path.empty:
            continue

        target_time = None
        target_price = None
        target_bar_offset = None
        for j, bar in path.iterrows():
            if side == "LONG":
                target = bar["executable_keltner_upper"]
                hit = pd.notna(target) and float(bar["high"]) >= float(target)
                if hit:
                    target_time = pd.Timestamp(bar["timestamp"])
                    target_price = max(float(bar["open"]), float(target))
                    target_bar_offset = int((target_time - entry_time) / pd.Timedelta(hours=1)) + 1
                    break
            else:
                target = bar["executable_keltner_lower"]
                hit = pd.notna(target) and float(bar["low"]) <= float(target)
                if hit:
                    target_time = pd.Timestamp(bar["timestamp"])
                    target_price = min(float(bar["open"]), float(target))
                    target_bar_offset = int((target_time - entry_time) / pd.Timedelta(hours=1)) + 1
                    break

        horizon_end = target_time if target_time is not None else pd.Timestamp(path.iloc[-1]["timestamp"])
        horizon = path[path["timestamp"] <= horizon_end]

        if side == "LONG":
            best_price = float(horizon["high"].max())
            worst_price = float(horizon["low"].min())
        else:
            best_price = float(horizon["low"].min())
            worst_price = float(horizon["high"].max())

        mfe_pct = _pct(side, entry, best_price)
        mae_pct = -abs(_pct(side, entry, worst_price))
        counterfactual_net = (
            _pct(side, entry, float(target_price)) - cost if target_price is not None else None
        )

        stop_bar_number = int((stop_time - entry_time) / pd.Timedelta(hours=1)) + 1
        hours_after_stop = (
            float((target_time - stop_time) / pd.Timedelta(hours=1)) if target_time is not None else None
        )

        rows.append(
            {
                "side": side,
                "entry_time": entry_time,
                "entry_price": entry,
                "entry_atr": float(trade["entry_atr"]),
                "stop_time": stop_time,
                "stop_price": float(trade["exit_price"]),
                "stop_net_return_pct": float(trade["net_return_pct"]),
                "stop_bar_number": stop_bar_number,
                "later_keltner_target_hit": target_time is not None,
                "counterfactual_target_time": target_time,
                "hours_after_stop_to_target": hours_after_stop,
                "counterfactual_target_price": target_price,
                "counterfactual_target_net_pct": counterfactual_net,
                "counterfactual_mfe_pct": mfe_pct,
                "counterfactual_mae_pct": mae_pct,
                "counterfactual_bars_to_target_or_end": target_bar_offset if target_bar_offset is not None else len(horizon),
            }
        )

    return pd.DataFrame(rows)


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
    stopped = run_stop_window(h1, eval_start, eval_end, cfg)
    diagnostic = diagnose_stopped_trades(h1, stopped, eval_end, cfg)

    if len(diagnostic) != 11:
        raise AssertionError(f"Expected 11 August ATR2 stops, got {len(diagnostic)}")

    later_hit = diagnostic["later_keltner_target_hit"]
    hit_count = int(later_hit.sum())
    no_hit_count = int((~later_hit).sum())
    recovered = diagnostic[later_hit]

    summary = {
        "diagnostic": "ichimoku-keltner-1h-atr2-stop-path-v1",
        "symbol": symbol,
        "evaluation_start_utc": eval_start.isoformat(),
        "evaluation_end_utc": eval_end.isoformat(),
        "observed_atr2_stops": int(len(diagnostic)),
        "later_keltner_target_hit": hit_count,
        "no_later_keltner_target_hit_before_eval_end": no_hit_count,
        "later_target_hit_rate_pct": float(hit_count / len(diagnostic) * 100.0),
        "median_hours_after_stop_to_target": (
            float(recovered["hours_after_stop_to_target"].median()) if not recovered.empty else None
        ),
        "median_counterfactual_mae_pct_when_later_target_hit": (
            float(recovered["counterfactual_mae_pct"].median()) if not recovered.empty else None
        ),
        "worst_counterfactual_mae_pct_when_later_target_hit": (
            float(recovered["counterfactual_mae_pct"].min()) if not recovered.empty else None
        ),
        "interpretation_rule": (
            "Later target hit is not labelled a false stop automatically; evaluate the adverse excursion and time required before recovery."
        ),
    }

    out = Path("artifacts/ichimoku-keltner-1h-stop-diagnostic")
    out.mkdir(parents=True, exist_ok=True)
    diagnostic.to_csv(out / "stopped_trade_counterfactuals.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str))
    print("\nSTOP COUNTERFACTUALS\n" + diagnostic.to_string(index=False))


if __name__ == "__main__":
    main()
