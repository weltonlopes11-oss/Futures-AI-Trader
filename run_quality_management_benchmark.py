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
from backtest.entry_quality import apply_entry_quality
from backtest.graphical_context import GraphicalContext, GraphicalContextConfig
from backtest.managed_exit_backtest import ManagedExitBacktest
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
    start = datetime.fromisoformat(
        os.getenv("BACKTEST_START_UTC", "2026-09-01T00:00:00+00:00").replace("Z", "+00:00")
    )
    end = datetime.fromisoformat(
        os.getenv("BACKTEST_END_UTC", "2026-09-15T00:00:00+00:00").replace("Z", "+00:00")
    )
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
    keltner_entries = apply_graphical_battery(core, "keltner")
    quality_entries = apply_entry_quality(core)

    fee = float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4"))
    slippage = float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1"))
    structure_engine = StructuralExitBacktest(
        fee_bps_per_side=fee,
        slippage_bps_per_side=slippage,
        structure_lookback=5,
    )
    managed_engine = ManagedExitBacktest(
        fee_bps_per_side=fee,
        slippage_bps_per_side=slippage,
        structure_lookback=5,
    )

    modes = [
        ("control_keltner_structure", keltner_entries, structure_engine),
        ("quality_entry_structure", quality_entries, structure_engine),
        ("keltner_managed_exit", keltner_entries, managed_engine),
        ("quality_entry_managed_exit", quality_entries, managed_engine),
    ]

    out = Path("artifacts/quality-management-benchmark")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for rr in (1.0, 1.5, 2.0, 3.0):
        for mode, decisions, engine in modes:
            trades = engine.run(decisions, decisions[["timestamp", "decision"]], rr=rr)
            met = engine.metrics(trades)
            if trades.empty:
                outcomes = {"structure_exits": 0, "breakeven_exits": 0, "keltner_middle_exits": 0}
            else:
                outcomes = {
                    "structure_exits": int((trades["outcome"] == "STRUCTURE_EXIT").sum()),
                    "breakeven_exits": int(trades["outcome"].astype(str).str.startswith("BREAKEVEN").sum()),
                    "keltner_middle_exits": int((trades["outcome"] == "KELTNER_MIDDLE_EXIT").sum()),
                }
            met.update({"mode": mode, "rr": rr, **outcomes})
            rows.append(met)
            trades.to_csv(out / f"trades_{mode}_rr_{rr}.csv", index=False)

    results = pd.DataFrame(rows)[
        [
            "mode", "rr", "trades", "win_rate_pct", "net_return_pct",
            "profit_factor", "expectancy_pct", "max_drawdown_pct", "payoff",
            "long", "short", "structure_exits", "breakeven_exits", "keltner_middle_exits",
        ]
    ]
    results.to_csv(out / "results.csv", index=False)

    manifest = {
        "benchmark": "quality-management-v1",
        "frozen_base": "Binance OI > 0 + relative funding-z + CVD + relative premium-z",
        "entry_control": "frozen Keltner battery 1",
        "entry_candidate": {
            "mandatory": "Keltner confirmation and abs(keltner_position) <= 1 outer-band anti-chase guard",
            "score_components": ["DMI direction agrees", "ADX rising", "correct side of UTC session VWAP"],
            "pass_rule": "at least 2 of 3 confirmations",
            "note": "thresholds predeclared before benchmark and not selected from PnL",
        },
        "exit_control": "1 ATR stop + fixed target + adverse 5-bar structural close break",
        "exit_candidate": {
            "break_even": "arm from next candle after a completed candle closes >= +1R",
            "keltner": "after break-even is active, exit at close on loss of Keltner middle",
            "priority": "active stop/target intrabar, then structure close exit, then Keltner-middle close exit",
        },
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
