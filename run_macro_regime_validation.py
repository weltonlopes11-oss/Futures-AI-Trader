from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.anti_extension import AntiExtensionConfig, AntiExtensionFilter
from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.binance_funding_rate import BinanceFundingRateLoader
from backtest.binance_metrics_data_vision import BinanceMetricsDataVisionLoader
from backtest.binance_positioning_context import BinancePositioningContextLoader
from backtest.cvd import CumulativeVolumeDelta
from backtest.graphical_context import GraphicalContext, GraphicalContextConfig
from backtest.structural_exit_backtest import StructuralExitBacktest
from run_90d_frozen_candidate import fetch_funding_history, longest_losing_streak
from run_graphical_benchmark import apply_graphical_battery
from run_operational_benchmark import atr, build_decisions, causal_context, causal_event_zscore, causal_relative_zscore, require_coverage, trend


def macro_regime(frame: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    x = frame.copy()
    x["ema_fast"] = x["close"].ewm(span=fast, adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=slow, adjust=False).mean()
    x["regime"] = "NEUTRAL"
    x.loc[(x["ema_fast"] > x["ema_slow"]) & (x["close"] > x["ema_fast"]), "regime"] = "BULL"
    x.loc[(x["ema_fast"] < x["ema_slow"]) & (x["close"] < x["ema_fast"]), "regime"] = "BEAR"
    x["available_at"] = pd.to_datetime(x["close_time"], errors="coerce")
    return x[["available_at", "regime"]].dropna().sort_values("available_at")


def align_regime(signal: pd.DataFrame, higher: pd.DataFrame, label: str) -> pd.DataFrame:
    h = macro_regime(higher).rename(columns={"regime": label})
    return pd.merge_asof(
        signal.sort_values("timestamp"),
        h,
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    ).drop(columns=["available_at"])


def apply_regime(decisions: pd.DataFrame, mode: str) -> pd.DataFrame:
    x = decisions.copy()
    if mode == "control":
        return x
    if mode == "daily_only":
        bull_ok = x["regime_1d"] == "BULL"
        bear_ok = x["regime_1d"] == "BEAR"
    elif mode == "weekly_daily":
        bull_ok = (x["regime_1w"] == "BULL") & (x["regime_1d"] == "BULL")
        bear_ok = (x["regime_1w"] == "BEAR") & (x["regime_1d"] == "BEAR")
    else:
        raise ValueError(mode)

    allow = ((x["decision"] == "LONG") & bull_ok) | ((x["decision"] == "SHORT") & bear_ok)
    x.loc[(x["decision"] != "NO_TRADE") & ~allow, "decision"] = "NO_TRADE"
    return x


def main():
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    eval_start = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_START_UTC", "2026-06-17T00:00:00+00:00").replace("Z", "+00:00"))
    eval_end = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_END_UTC", "2026-09-15T00:00:00+00:00").replace("Z", "+00:00"))
    if eval_start.tzinfo is None: eval_start = eval_start.replace(tzinfo=timezone.utc)
    if eval_end.tzinfo is None: eval_end = eval_end.replace(tzinfo=timezone.utc)

    warmup_days = int(os.getenv("BACKTEST_WARMUP_DAYS", "180"))
    fetch_start = eval_start - timedelta(days=warmup_days)

    prices = BinanceDataVisionLoader()
    metrics_loader = BinanceMetricsDataVisionLoader()
    funding_loader = BinanceFundingRateLoader()
    positioning_loader = BinancePositioningContextLoader()
    cvd_builder = CumulativeVolumeDelta(fast=12, slow=26)
    graphical = GraphicalContext(GraphicalContextConfig())

    m15 = prices.fetch_window(symbol, "15m", fetch_start, eval_end)
    h1 = prices.fetch_window(symbol, "1h", fetch_start, eval_end)
    h4 = prices.fetch_window(symbol, "4h", fetch_start, eval_end)
    d1 = prices.fetch_window(symbol, "1d", fetch_start, eval_end)
    w1 = prices.fetch_window(symbol, "1w", fetch_start, eval_end)
    oi = metrics_loader.fetch_window(symbol, fetch_start, eval_end)
    funding = fetch_funding_history(symbol, fetch_start, eval_end, funding_loader)
    funding["funding_z"] = causal_event_zscore(funding["funding_rate"])
    premium = positioning_loader.fetch_premium_index(symbol, fetch_start, eval_end, interval="15m")

    m15["atr"] = atr(m15)
    m15["signal_15m"] = trend(m15)
    m15 = cvd_builder.enrich(m15)
    m15 = graphical.enrich(m15)

    context = causal_context(m15, h1, "trend_1h")
    context = causal_context(context, h4, "trend_4h")
    context = align_regime(context, d1, "regime_1d")
    context = align_regime(context, w1, "regime_1w")
    context = metrics_loader.align_causally(context, oi)
    context = funding_loader.align_causally(context, funding)
    context = positioning_loader.align_causally(context, premium)
    context["premium_z"] = causal_relative_zscore(context["premium_close"])

    core = build_decisions(
        context,
        oi_mode="binance",
        funding_mode="relative",
        use_cvd=True,
        use_premium=True,
        use_positioning=False,
        use_leverage_stress=False,
    )
    keltner = apply_graphical_battery(core, "keltner")
    anti = AntiExtensionFilter(AntiExtensionConfig(max_vwap_distance_atr=2.0, max_keltner_position=0.75))
    base = anti.apply(keltner, "keltner_only")

    start_naive = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end_naive = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")
    base = base[(base["timestamp"] >= start_naive) & (base["timestamp"] < end_naive)].reset_index(drop=True)

    engine = StructuralExitBacktest(
        fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")),
        slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")),
        structure_lookback=5,
    )

    out = Path("artifacts/macro-regime-validation")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for mode in ["control", "daily_only", "weekly_daily"]:
        d = apply_regime(base, mode)
        trades = engine.run(d, d[["timestamp", "decision"]], rr=2.0)
        met = engine.metrics(trades)
        met["longest_losing_streak"] = longest_losing_streak(trades)
        met["mode"] = mode
        met["allowed_long_signals"] = int((d["decision"] == "LONG").sum())
        met["allowed_short_signals"] = int((d["decision"] == "SHORT").sum())
        rows.append(met)
        trades.to_csv(out / f"trades_{mode}.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(out / "metrics.csv", index=False)

    manifest = {
        "benchmark": "macro-regime-validation-v1",
        "symbol": symbol,
        "evaluation_start_utc": eval_start.isoformat(),
        "evaluation_end_utc": eval_end.isoformat(),
        "macro_rule": {
            "bull": "EMA20 > EMA50 and close > EMA20",
            "bear": "EMA20 < EMA50 and close < EMA20",
            "neutral": "otherwise",
            "daily_only": "LONG only in 1D bull; SHORT only in 1D bear; neutral blocks",
            "weekly_daily": "LONG only when 1W+1D bull; SHORT only when 1W+1D bear; disagreement blocks",
        },
        "strategy_unchanged": "core + Keltner + anti-extension 0.75 + 1ATR stop + 2R + 5-bar structural exit",
        "anti_overfit": "No original strategy parameter changed; only predeclared macro permission filters compared.",
        "results": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
