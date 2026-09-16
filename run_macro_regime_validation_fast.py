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
from backtest.binance_monthly_klines import BinanceMonthlyKlineLoader
from backtest.binance_positioning_context import BinancePositioningContextLoader
from backtest.cvd import CumulativeVolumeDelta
from backtest.graphical_context import GraphicalContext, GraphicalContextConfig
from backtest.structural_exit_backtest import StructuralExitBacktest
from run_90d_frozen_candidate import fetch_funding_history, longest_losing_streak
from run_graphical_benchmark import apply_graphical_battery
from run_macro_regime_validation import align_regime, apply_regime
from run_operational_benchmark import atr, build_decisions, causal_context, causal_event_zscore, causal_relative_zscore, trend


def main():
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    eval_start = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_START_UTC", "2026-06-17T00:00:00+00:00").replace("Z", "+00:00"))
    eval_end = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_END_UTC", "2026-09-15T00:00:00+00:00").replace("Z", "+00:00"))
    if eval_start.tzinfo is None: eval_start = eval_start.replace(tzinfo=timezone.utc)
    if eval_end.tzinfo is None: eval_end = eval_end.replace(tzinfo=timezone.utc)
    strategy_fetch_start = eval_start - timedelta(days=int(os.getenv("BACKTEST_STRATEGY_WARMUP_DAYS", "14")))
    macro_fetch_start = eval_start - timedelta(days=int(os.getenv("BACKTEST_MACRO_WARMUP_DAYS", "420")))

    prices = BinanceDataVisionLoader()
    monthly = BinanceMonthlyKlineLoader()
    metrics_loader = BinanceMetricsDataVisionLoader()
    funding_loader = BinanceFundingRateLoader()
    positioning_loader = BinancePositioningContextLoader()

    m15 = prices.fetch_window(symbol, "15m", strategy_fetch_start, eval_end)
    h1 = prices.fetch_window(symbol, "1h", strategy_fetch_start, eval_end)
    h4 = prices.fetch_window(symbol, "4h", strategy_fetch_start, eval_end)
    oi = metrics_loader.fetch_window(symbol, strategy_fetch_start, eval_end)
    funding = fetch_funding_history(symbol, strategy_fetch_start, eval_end, funding_loader)
    funding["funding_z"] = causal_event_zscore(funding["funding_rate"])
    premium = positioning_loader.fetch_premium_index(symbol, strategy_fetch_start, eval_end, interval="15m")
    d1 = monthly.fetch_window(symbol, "1d", macro_fetch_start, eval_end)
    w1 = monthly.fetch_window(symbol, "1w", macro_fetch_start, eval_end)

    m15["atr"] = atr(m15)
    m15["signal_15m"] = trend(m15)
    m15 = CumulativeVolumeDelta(fast=12, slow=26).enrich(m15)
    m15 = GraphicalContext(GraphicalContextConfig()).enrich(m15)

    context = causal_context(m15, h1, "trend_1h")
    context = causal_context(context, h4, "trend_4h")
    context = align_regime(context, d1, "regime_1d")
    context = align_regime(context, w1, "regime_1w")
    context = metrics_loader.align_causally(context, oi)
    context = funding_loader.align_causally(context, funding)
    context = positioning_loader.align_causally(context, premium)
    context["premium_z"] = causal_relative_zscore(context["premium_close"])

    core = build_decisions(context, oi_mode="binance", funding_mode="relative", use_cvd=True, use_premium=True)
    keltner = apply_graphical_battery(core, "keltner")
    base = AntiExtensionFilter(AntiExtensionConfig(max_vwap_distance_atr=2.0, max_keltner_position=0.75)).apply(keltner, "keltner_only")
    start_n = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    end_n = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")
    base = base[(base.timestamp >= start_n) & (base.timestamp < end_n)].reset_index(drop=True)

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
        met.update({
            "mode": mode,
            "longest_losing_streak": longest_losing_streak(trades),
            "trade_longs": int((trades.side == "LONG").sum()) if len(trades) else 0,
            "trade_shorts": int((trades.side == "SHORT").sum()) if len(trades) else 0,
        })
        rows.append(met)
        trades.to_csv(out / f"trades_{mode}.csv", index=False)

    control = rows[0]
    if int(control.get("trades", -1)) != 53 or abs(float(control.get("net_return_pct", 999)) - (-6.3570055599705855)) > 1e-6:
        raise RuntimeError(f"Control mismatch: {control}")

    manifest = {
        "benchmark": "macro-regime-validation-v1-fast",
        "evaluation_start_utc": eval_start.isoformat(),
        "evaluation_end_utc": eval_end.isoformat(),
        "macro_data_source": "official Binance Data Vision monthly 1d/1w klines",
        "macro_rule": "EMA20>EMA50 & close>EMA20=BULL; EMA20<EMA50 & close<EMA20=BEAR; else NEUTRAL; only completed 1D/1W candles are causal",
        "direction_rule": "daily_only: bull LONG only / bear SHORT only; weekly_daily: 1W+1D both bull LONG only / both bear SHORT only / disagreement no trade",
        "strategy_unchanged": "core + Keltner + anti-extension 0.75 + 1ATR stop + 2R + 5-bar structural exit",
        "results": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
