from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.binance_funding_rate import BinanceFundingRateLoader
from backtest.binance_metrics_data_vision import BinanceMetricsDataVisionLoader
from backtest.binance_positioning_context import BinancePositioningContextLoader
from backtest.cvd import CumulativeVolumeDelta
from backtest.graphical_context import GraphicalContext, GraphicalContextConfig
from backtest.price_structure_patterns import PriceStructureConfig, PriceStructurePatterns
from backtest.structural_exit_backtest import StructuralExitBacktest
from backtest.trade_quality_analytics import TradeQualityAnalytics
from run_graphical_benchmark import apply_graphical_battery
from run_operational_benchmark import (
    atr,
    build_decisions,
    causal_context,
    causal_event_zscore,
    causal_relative_zscore,
    require_coverage,
    trend,
)


def main():
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    start = datetime.fromisoformat(os.getenv("BACKTEST_START_UTC", "2026-09-01T00:00:00+00:00").replace("Z", "+00:00"))
    end = datetime.fromisoformat(os.getenv("BACKTEST_END_UTC", "2026-09-15T00:00:00+00:00").replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    prices = BinanceDataVisionLoader()
    metrics = BinanceMetricsDataVisionLoader()
    funding_loader = BinanceFundingRateLoader()
    positioning_loader = BinancePositioningContextLoader()
    cvd_builder = CumulativeVolumeDelta(fast=12, slow=26)
    graphical = GraphicalContext(GraphicalContextConfig())
    structure = PriceStructurePatterns(PriceStructureConfig())
    analytics = TradeQualityAnalytics()

    funding_snapshot = Path("research_snapshots/ETHUSDT_funding_2026-09-01_2026-09-15.csv")
    m15 = prices.fetch_window(symbol, "15m", start, end)
    h1 = prices.fetch_window(symbol, "1h", start, end)
    h4 = prices.fetch_window(symbol, "4h", start, end)
    oi = metrics.fetch_window(symbol, start, end)
    funding = funding_loader.fetch_window(symbol, start, end, snapshot_path=funding_snapshot)
    funding["funding_z"] = causal_event_zscore(funding["funding_rate"])
    premium = positioning_loader.fetch_premium_index(symbol, start, end, interval="15m")

    mins = int((end - start).total_seconds() // 60)
    require_coverage(m15, mins // 15, "15m")
    require_coverage(h1, mins // 60, "1h")
    require_coverage(h4, mins // 240, "4h")

    m15["atr"] = atr(m15)
    m15["signal_15m"] = trend(m15)
    m15 = cvd_builder.enrich(m15)
    m15 = graphical.enrich(m15)
    m15 = structure.enrich(m15)

    context = causal_context(m15, h1, "trend_1h")
    context = causal_context(context, h4, "trend_4h")
    context = metrics.align_causally(context, oi)
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

    fee = float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4"))
    slippage = float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1"))
    engine = StructuralExitBacktest(
        fee_bps_per_side=fee,
        slippage_bps_per_side=slippage,
        structure_lookback=5,
    )
    rr = 2.0
    trades = engine.run(keltner, keltner[["timestamp", "decision"]], rr=rr)
    enriched = analytics.enrich_trades(keltner, keltner, trades)
    summary = analytics.winner_loser_summary(enriched)

    out = Path("artifacts/trade-quality-analytics")
    out.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out / "trades_enriched_rr_2.csv", index=False)
    summary.to_csv(out / "winner_loser_summary_rr_2.csv", index=False)

    feature_counts = {}
    for col in ["entry_structure_regime", "entry_swing_high_state", "entry_swing_low_state"]:
        if col in enriched.columns:
            feature_counts[col] = enriched[col].fillna("NA").value_counts().to_dict()

    manifest = {
        "benchmark": "trade-quality-analytics-v1.1",
        "symbol": symbol,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "rr": rr,
        "entry": "frozen core + Keltner",
        "exit": "1 ATR stop + 2R target + adverse five-bar structural exit",
        "trades": int(len(enriched)),
        "winners": int(enriched["winner"].sum()) if len(enriched) else 0,
        "losers": int((~enriched["winner"]).sum()) if len(enriched) else 0,
        "mean_mfe_r": float(enriched["mfe_r"].mean()) if len(enriched) else 0.0,
        "median_mfe_r": float(enriched["mfe_r"].median()) if len(enriched) else 0.0,
        "mean_mae_r": float(enriched["mae_r"].mean()) if len(enriched) else 0.0,
        "median_mae_r": float(enriched["mae_r"].median()) if len(enriched) else 0.0,
        "reached_0_5r": int(enriched["reached_0_5r"].sum()) if len(enriched) else 0,
        "reached_1r": int(enriched["reached_1r"].sum()) if len(enriched) else 0,
        "reached_2r": int(enriched["reached_2r"].sum()) if len(enriched) else 0,
        "structural_entry_counts": feature_counts,
        "note": "Diagnostic only. Persistent swing states and event recency are causal and use only information known at signal_time. No thresholds or trading rules are promoted from this frozen September sample.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    print("\nWINNER / LOSER SUMMARY")
    print(summary.to_string(index=False))
    print("\nTRADE LEVEL")
    show = [c for c in [
        "signal_time", "side", "outcome", "net_return_pct", "mfe_r", "mae_r",
        "bars_to_mfe", "bars_to_mae", "trade_duration_bars",
        "entry_structure_regime", "entry_swing_high_state", "entry_swing_low_state",
        "entry_bars_since_bos_up", "entry_bars_since_bos_down",
        "entry_bars_since_choch_up", "entry_bars_since_choch_down",
        "entry_bars_since_double_top", "entry_bars_since_double_bottom",
        "entry_distance_to_swing_high_atr", "entry_distance_to_swing_low_atr",
        "entry_keltner_position", "entry_adx", "entry_adx_delta", "entry_vwap_distance_atr",
        "entry_oi_change_pct", "entry_funding_z", "entry_premium_z",
    ] if c in enriched.columns]
    print(enriched[show].to_string(index=False))


if __name__ == "__main__":
    main()
