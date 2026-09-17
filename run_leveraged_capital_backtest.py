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
from backtest.leveraged_capital import LeveragedCapitalConfig, LeveragedCapitalSimulator
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
    decisions = anti.apply(keltner, "keltner_only")

    fee = float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4"))
    slippage = float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1"))
    rr = 2.0
    trade_engine = StructuralExitBacktest(
        fee_bps_per_side=fee,
        slippage_bps_per_side=slippage,
        structure_lookback=5,
    )
    trades = trade_engine.run(decisions, decisions[["timestamp", "decision"]], rr=rr)

    capital_cfg = LeveragedCapitalConfig(
        initial_capital=float(os.getenv("CAPITAL_INITIAL_BRL", "500")),
        leverage=float(os.getenv("CAPITAL_LEVERAGE", "10")),
        margin_fraction=float(os.getenv("CAPITAL_MARGIN_FRACTION", "1")),
        maintenance_margin_rate=float(os.getenv("CAPITAL_MAINTENANCE_MARGIN_RATE", "0.005")),
    )
    simulator = LeveragedCapitalSimulator(capital_cfg)
    simulated = simulator.simulate(decisions, trades)
    summary = simulator.summary(simulated)

    out = Path("artifacts/leveraged-capital-backtest")
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "strategy_trades.csv", index=False)
    simulated.to_csv(out / "capital_curve_trade_by_trade.csv", index=False)

    manifest = {
        "benchmark": "leveraged-capital-v1",
        "symbol": symbol,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "strategy": "frozen core + Keltner confirmation + Keltner anti-extension 0.75 + 5-bar structural exit",
        "rr": rr,
        "fee_bps_per_side": fee,
        "slippage_bps_per_side": slippage,
        "initial_capital_brl": capital_cfg.initial_capital,
        "leverage": capital_cfg.leverage,
        "margin_fraction": capital_cfg.margin_fraction,
        "margin_mode_assumption": "isolated diagnostic; full configured margin fraction allocated per trade",
        "maintenance_margin_rate_assumption": capital_cfg.maintenance_margin_rate,
        "approx_liquidation_adverse_pct": simulator.approximate_liquidation_adverse_pct,
        "liquidation_note": "Approximation only. Exact Binance liquidation depends on mark price, maintenance tier, fees and account margin mode.",
        "funding_cashflow_note": "Funding rate is used as a signal feature but explicit funding cashflows are not yet debited/credited in trade PnL.",
        **summary,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(json.dumps(manifest, indent=2, default=str))
    if not simulated.empty:
        show = [
            "trade_number", "signal_time", "side", "outcome", "net_return_pct",
            "capital_before", "notional", "leveraged_pnl", "capital_after",
            "margin_return_pct", "equity_drawdown_pct", "max_adverse_price_pct",
            "approx_liquidation_adverse_pct", "liquidation_buffer_pct", "approx_liquidation_hit",
        ]
        print("\nTRADE BY TRADE")
        print(simulated[show].to_string(index=False))


if __name__ == "__main__":
    main()
