from __future__ import annotations

import json
import os
from datetime import datetime, timezone
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
    anti = AntiExtensionFilter(AntiExtensionConfig(max_vwap_distance_atr=2.0, max_keltner_position=0.75))

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

    out = Path("artifacts/anti-extension-benchmark")
    out.mkdir(parents=True, exist_ok=True)

    modes = ["control", "vwap_only", "keltner_only", "combined_either", "combined_both"]
    rows = []
    for rr in (1.0, 1.5, 2.0, 3.0):
        for mode in modes:
            decisions = anti.apply(keltner, mode)
            rejected = int(decisions["anti_extension_rejected"].sum())
            trades = engine.run(decisions, decisions[["timestamp", "decision"]], rr=rr)
            met = engine.metrics(trades)
            met.update({"mode": mode, "rr": rr, "signals_rejected": rejected})
            rows.append(met)
            trades.to_csv(out / f"trades_{mode}_rr_{rr}.csv", index=False)

    results = pd.DataFrame(rows)[
        [
            "mode", "rr", "signals_rejected", "trades", "win_rate_pct", "net_return_pct",
            "profit_factor", "expectancy_pct", "max_drawdown_pct", "payoff", "long", "short",
        ]
    ]
    results.to_csv(out / "results.csv", index=False)

    manifest = {
        "benchmark": "anti-extension-v1",
        "status": "diagnostic-research-candidate",
        "base_entry": "frozen core + Keltner",
        "base_exit": "1 ATR stop + R target + adverse five-bar structural exit",
        "primary_rr": 2.0,
        "vwap_threshold_atr": anti.config.max_vwap_distance_atr,
        "keltner_directional_threshold": anti.config.max_keltner_position,
        "directional_definition": "LONG uses raw extension; SHORT uses sign-inverted extension so positive always means stretched in trade direction",
        "modes": {
            "control": "no anti-extension rejection",
            "vwap_only": "reject if directional VWAP distance > 2 ATR",
            "keltner_only": "reject if directional Keltner position > 0.75",
            "combined_either": "reject if either threshold is exceeded",
            "combined_both": "reject only if both thresholds are exceeded",
        },
        "anti_overfit": "Thresholds were declared before this benchmark result. Do not select or retune a mode from the Sep 1-15 PnL alone.",
        "symbol": symbol,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "fee_bps_per_side": fee,
        "slippage_bps_per_side": slippage,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
