from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.anti_extension import AntiExtensionConfig, AntiExtensionFilter
from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.binance_funding_rate import BinanceFundingRateLoader
from backtest.binance_metrics_data_vision import BinanceMetricsDataVisionLoader
from backtest.binance_monthly_klines import BinanceMonthlyKlineLoader
from backtest.binance_positioning_context import BinancePositioningContextLoader
from backtest.cvd import CumulativeVolumeDelta
from backtest.graphical_context import GraphicalContext, GraphicalContextConfig
from backtest.structural_exit_backtest import StructuralExitBacktest
from backtest.trade_quality_analytics import TradeQualityAnalytics
from run_90d_frozen_candidate import fetch_funding_history
from run_graphical_benchmark import apply_graphical_battery
from run_macro_regime_validation import apply_regime
from run_macro_regime_validation_fast import load_daily_macro
from run_operational_benchmark import atr, build_decisions, causal_context, causal_event_zscore, causal_relative_zscore, trend


def daily_features(d1: pd.DataFrame) -> pd.DataFrame:
    x = d1.copy().sort_values("timestamp").reset_index(drop=True)
    x["daily_ema20"] = x["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    x["daily_ema50"] = x["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    valid = x["daily_ema20"].notna() & x["daily_ema50"].notna()
    x["regime_1d"] = "NEUTRAL"
    x.loc[valid & (x["daily_ema20"] > x["daily_ema50"]) & (x["close"] > x["daily_ema20"]), "regime_1d"] = "BULL"
    x.loc[valid & (x["daily_ema20"] < x["daily_ema50"]) & (x["close"] < x["daily_ema20"]), "regime_1d"] = "BEAR"
    x["daily_close"] = x["close"]
    x["daily_ema_gap_pct"] = (x["daily_ema20"] - x["daily_ema50"]) / x["daily_close"] * 100.0
    x["daily_close_vs_ema20_pct"] = (x["daily_close"] - x["daily_ema20"]) / x["daily_ema20"] * 100.0
    x["daily_ema20_slope_1d_pct"] = x["daily_ema20"].pct_change(1) * 100.0
    x["daily_ema20_slope_5d_pct"] = x["daily_ema20"].pct_change(5) * 100.0
    x["daily_available_at"] = pd.to_datetime(x["close_time"], errors="coerce")
    cols = [
        "daily_available_at", "regime_1d", "daily_close", "daily_ema20", "daily_ema50",
        "daily_ema_gap_pct", "daily_close_vs_ema20_pct", "daily_ema20_slope_1d_pct", "daily_ema20_slope_5d_pct",
    ]
    return x[cols].dropna(subset=["daily_available_at"]).sort_values("daily_available_at")


def align_daily(signal: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(
        signal.sort_values("timestamp"), daily_features(d1), left_on="timestamp", right_on="daily_available_at", direction="backward"
    ).drop(columns=["daily_available_at"])


def directional(row: pd.Series, col: str) -> float:
    v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    if pd.isna(v):
        return np.nan
    return float(v) if str(row["side"]).upper() == "LONG" else -float(v)


def subset_metrics(engine: StructuralExitBacktest, trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    m = engine.metrics(trades)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v) for k, v in m.items()}


def main():
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    eval_start = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_START_UTC", "2026-06-17T00:00:00+00:00").replace("Z", "+00:00"))
    eval_end = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_END_UTC", "2026-09-15T00:00:00+00:00").replace("Z", "+00:00"))
    if eval_start.tzinfo is None: eval_start = eval_start.replace(tzinfo=timezone.utc)
    if eval_end.tzinfo is None: eval_end = eval_end.replace(tzinfo=timezone.utc)
    strategy_start = eval_start - timedelta(days=14)
    macro_start = eval_start - timedelta(days=120)

    prices = BinanceDataVisionLoader(); monthly = BinanceMonthlyKlineLoader()
    metrics_loader = BinanceMetricsDataVisionLoader(); funding_loader = BinanceFundingRateLoader()
    positioning_loader = BinancePositioningContextLoader()

    m15 = prices.fetch_window(symbol, "15m", strategy_start, eval_end)
    h1 = prices.fetch_window(symbol, "1h", strategy_start, eval_end)
    h4 = prices.fetch_window(symbol, "4h", strategy_start, eval_end)
    oi = metrics_loader.fetch_window(symbol, strategy_start, eval_end)
    funding = fetch_funding_history(symbol, strategy_start, eval_end, funding_loader)
    funding["funding_z"] = causal_event_zscore(funding["funding_rate"])
    premium = positioning_loader.fetch_premium_index(symbol, strategy_start, eval_end, interval="15m")
    d1 = load_daily_macro(symbol, macro_start, eval_end, monthly, prices)

    m15["atr"] = atr(m15); m15["signal_15m"] = trend(m15)
    m15 = CumulativeVolumeDelta(fast=12, slow=26).enrich(m15)
    m15 = GraphicalContext(GraphicalContextConfig()).enrich(m15)
    context = causal_context(m15, h1, "trend_1h")
    context = causal_context(context, h4, "trend_4h")
    context = align_daily(context, d1)
    context = metrics_loader.align_causally(context, oi)
    context = funding_loader.align_causally(context, funding)
    context = positioning_loader.align_causally(context, premium)
    context["premium_z"] = causal_relative_zscore(context["premium_close"])

    core = build_decisions(context, oi_mode="binance", funding_mode="relative", use_cvd=True, use_premium=True)
    keltner = apply_graphical_battery(core, "keltner")
    base = AntiExtensionFilter(AntiExtensionConfig(max_vwap_distance_atr=2.0, max_keltner_position=0.75)).apply(keltner, "keltner_only")
    start_n = eval_start.astimezone(timezone.utc).replace(tzinfo=None); end_n = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")
    base = base[(base["timestamp"] >= start_n) & (base["timestamp"] < end_n)].reset_index(drop=True)
    decisions = apply_regime(base, "daily_only")

    engine = StructuralExitBacktest(fee_bps_per_side=4.0, slippage_bps_per_side=1.0, structure_lookback=5)
    trades = engine.run(decisions, decisions[["timestamp", "decision"]], rr=2.0)
    if len(trades) != 22:
        raise RuntimeError(f"Expected 22 daily-regime trades, got {len(trades)}")

    enriched = TradeQualityAnalytics().enrich_trades(m15, decisions, trades)
    snapshots = decisions.set_index(pd.to_datetime(decisions["timestamp"], utc=True))
    wanted = [
        "signal_15m", "trend_1h", "trend_4h", "regime_1d", "daily_ema_gap_pct", "daily_close_vs_ema20_pct",
        "daily_ema20_slope_1d_pct", "daily_ema20_slope_5d_pct", "keltner_position", "keltner_width_pct", "adx",
        "plus_di", "minus_di", "vwap_distance_atr", "open_interest_change_pct", "funding_z", "premium_z",
        "cvd_fast", "cvd_slow", "cvd_signal", "atr", "close",
    ]
    for i, tr in enriched.iterrows():
        ts = pd.to_datetime(tr["signal_time"], utc=True)
        snap = snapshots.loc[ts]
        if isinstance(snap, pd.DataFrame): snap = snap.iloc[-1]
        for col in wanted:
            if col in snap.index: enriched.loc[i, f"entry_{col}"] = snap[col]

    enriched["month"] = pd.to_datetime(enriched["signal_time"]).dt.strftime("%Y-%m")
    enriched["winner"] = enriched["net_return_pct"] > 0
    enriched["directional_keltner"] = enriched.apply(lambda r: directional(r, "entry_keltner_position"), axis=1)
    enriched["directional_vwap_atr"] = enriched.apply(lambda r: directional(r, "entry_vwap_distance_atr"), axis=1)
    enriched["directional_daily_ema_gap_pct"] = enriched.apply(lambda r: directional(r, "entry_daily_ema_gap_pct"), axis=1)
    enriched["directional_daily_close_vs_ema20_pct"] = enriched.apply(lambda r: directional(r, "entry_daily_close_vs_ema20_pct"), axis=1)
    enriched["directional_daily_ema20_slope_1d_pct"] = enriched.apply(lambda r: directional(r, "entry_daily_ema20_slope_1d_pct"), axis=1)
    enriched["directional_daily_ema20_slope_5d_pct"] = enriched.apply(lambda r: directional(r, "entry_daily_ema20_slope_5d_pct"), axis=1)

    enriched = enriched.sort_values("signal_time").reset_index(drop=True)
    enriched["hours_since_prior_trade"] = pd.to_datetime(enriched["signal_time"]).diff().dt.total_seconds() / 3600.0
    enriched["prior_trade_was_loss"] = (~enriched["winner"].shift(1).fillna(True)).astype(bool)
    enriched["rapid_reentry_24h"] = enriched["hours_since_prior_trade"].le(24.0).fillna(False)

    out = Path("artifacts/daily-regime-trade-investigation"); out.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out / "trades_enriched.csv", index=False)

    breakdown_rows = []
    for dimension in ["month", "side", "outcome"]:
        for value, grp in enriched.groupby(dimension, dropna=False):
            row = {"dimension": dimension, "value": str(value)}
            row.update(subset_metrics(engine, grp))
            breakdown_rows.append(row)
    breakdown = pd.DataFrame(breakdown_rows); breakdown.to_csv(out / "breakdowns.csv", index=False)

    numeric = [
        "net_return_pct", "mfe_r", "mae_r", "trade_duration_bars", "directional_keltner", "directional_vwap_atr",
        "entry_adx", "entry_open_interest_change_pct", "entry_funding_z", "entry_premium_z",
        "directional_daily_ema_gap_pct", "directional_daily_close_vs_ema20_pct",
        "directional_daily_ema20_slope_1d_pct", "directional_daily_ema20_slope_5d_pct",
    ]
    summary_rows = []
    for keys, grp in enriched.groupby(["side", "winner"], dropna=False):
        row = {"side": keys[0], "group": "winner" if keys[1] else "loser", "trades": len(grp)}
        for col in numeric:
            vals = pd.to_numeric(grp[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else None
            row[f"{col}_median"] = float(vals.median()) if vals.notna().any() else None
        summary_rows.append(row)
    feature_summary = pd.DataFrame(summary_rows); feature_summary.to_csv(out / "winner_loser_features.csv", index=False)

    manifest = {
        "investigation": "daily-regime-22-trades-v1",
        "purpose": "diagnosis only; no thresholds optimized and no strategy rule changed",
        "evaluation": [eval_start.isoformat(), eval_end.isoformat()],
        "control_guardrail": {"expected_trades": 22, "actual_trades": len(trades)},
        "overall": subset_metrics(engine, enriched),
        "counts": {
            "winners": int(enriched["winner"].sum()), "losers": int((~enriched["winner"]).sum()),
            "rapid_reentries_24h": int(enriched["rapid_reentry_24h"].sum()),
            "rapid_reentry_losses": int((enriched["rapid_reentry_24h"] & ~enriched["winner"]).sum()),
            "stop": int((enriched["outcome"] == "STOP").sum()), "target": int((enriched["outcome"] == "TARGET").sum()),
            "structure_exit": int((enriched["outcome"] == "STRUCTURE_EXIT").sum()),
        },
        "mfe_mae_note": "Exploratory OHLC excursion through exit candle; do not use as exact intrabar execution ordering for STOP/TARGET exits.",
        "artifacts": ["trades_enriched.csv", "breakdowns.csv", "winner_loser_features.csv"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    print("\nBREAKDOWNS\n", breakdown.to_string(index=False))
    print("\nWINNER/LOSER FEATURES\n", feature_summary.to_string(index=False))


if __name__ == "__main__":
    main()
