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
from backtest.operational_backtest import OperationalBacktest
from backtest.pivot_structure_exit_backtest import PivotStructureExitBacktest
from backtest.price_structure_patterns import PriceStructureConfig, PriceStructurePatterns
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


def apply_structure_entry(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = frame.copy()
    decision = out["decision"].astype(str).str.upper()
    long = decision == "LONG"
    short = decision == "SHORT"

    high_state = out["swing_high_class"].replace("NONE", pd.NA).ffill()
    low_state = out["swing_low_class"].replace("NONE", pd.NA).ffill()

    if mode == "control":
        keep = long | short
    elif mode == "swing_label_alignment":
        keep = (long & ((low_state == "HL") | (high_state == "HH"))) | (
            short & ((high_state == "LH") | (low_state == "LL"))
        )
    elif mode == "structure_regime":
        keep = (long & (out["structure_regime"] == "BULLISH")) | (
            short & (out["structure_regime"] == "BEARISH")
        )
    elif mode == "bos_choch_event":
        keep = (long & (out["bos_up"] | out["choch_up"])) | (
            short & (out["bos_down"] | out["choch_down"])
        )
    elif mode == "double_reversal_event":
        keep = (long & out["double_bottom"]) | (short & out["double_top"])
    elif mode == "any_structural_confirmation":
        keep = (
            long
            & (
                (out["structure_regime"] == "BULLISH")
                | out["bos_up"]
                | out["choch_up"]
                | out["double_bottom"]
            )
        ) | (
            short
            & (
                (out["structure_regime"] == "BEARISH")
                | out["bos_down"]
                | out["choch_down"]
                | out["double_top"]
            )
        )
    else:
        raise ValueError(f"unknown structure mode: {mode}")

    out.loc[~keep, "decision"] = "NO_TRADE"
    return out


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
    structure_exit = StructuralExitBacktest(fee_bps_per_side=fee, slippage_bps_per_side=slippage, structure_lookback=5)
    pivot_exit = PivotStructureExitBacktest(fee_bps_per_side=fee, slippage_bps_per_side=slippage)

    out = Path("artifacts/price-structure-benchmark")
    out.mkdir(parents=True, exist_ok=True)

    entry_modes = [
        "control",
        "swing_label_alignment",
        "structure_regime",
        "bos_choch_event",
        "double_reversal_event",
        "any_structural_confirmation",
    ]
    rows = []
    for rr in (1.0, 1.5, 2.0, 3.0):
        for entry_mode in entry_modes:
            decisions = apply_structure_entry(keltner, entry_mode)
            trades = structure_exit.run(decisions, decisions[["timestamp", "decision"]], rr=rr)
            met = structure_exit.metrics(trades)
            met.update({"entry_mode": entry_mode, "exit_mode": "five_bar", "rr": rr})
            rows.append(met)
            trades.to_csv(out / f"trades_{entry_mode}_five_bar_rr_{rr}.csv", index=False)

        # Exit ablation uses exactly the unchanged Keltner control entries.
        trades = pivot_exit.run(keltner, keltner[["timestamp", "decision"]], rr=rr)
        met = pivot_exit.metrics(trades)
        met.update({"entry_mode": "control", "exit_mode": "confirmed_pivot_invalidation", "rr": rr})
        rows.append(met)
        trades.to_csv(out / f"trades_control_pivot_exit_rr_{rr}.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)

    diagnostics = {
        "confirmed_pivot_highs": int(m15["pivot_high_confirmed"].sum()),
        "confirmed_pivot_lows": int(m15["pivot_low_confirmed"].sum()),
        "HH": int((m15["swing_high_class"] == "HH").sum()),
        "LH": int((m15["swing_high_class"] == "LH").sum()),
        "HL": int((m15["swing_low_class"] == "HL").sum()),
        "LL": int((m15["swing_low_class"] == "LL").sum()),
        "bos_up": int(m15["bos_up"].sum()),
        "bos_down": int(m15["bos_down"].sum()),
        "choch_up": int(m15["choch_up"].sum()),
        "choch_down": int(m15["choch_down"].sum()),
        "double_top": int(m15["double_top"].sum()),
        "double_bottom": int(m15["double_bottom"].sum()),
    }
    manifest = {
        "benchmark": "price-structure-v1",
        "frozen_base": "core + Keltner entry; five-bar adverse structural exit control",
        "pivot_span": 3,
        "pivot_confirmation_delay_minutes": 45,
        "double_tolerance_atr": 0.25,
        "double_min_separation_bars": 4,
        "bos_choch": "close breaks latest previously confirmed swing level; CHOCH is a break against prior structural regime",
        "diagnostics": diagnostics,
        "symbol": symbol,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "fee_bps_per_side": fee,
        "slippage_bps_per_side": slippage,
        "note": "All parameters predeclared before benchmark; no result-driven tuning.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
