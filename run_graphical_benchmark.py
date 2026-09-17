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
from run_operational_benchmark import (
    atr,
    build_decisions,
    causal_context,
    causal_event_zscore,
    causal_relative_zscore,
    require_coverage,
    trend,
)


def apply_graphical_battery(core: pd.DataFrame, battery: str) -> pd.DataFrame:
    c = core.copy()
    base = c["decision"].copy()

    required = {
        "keltner": ["keltner_long_confirm", "keltner_short_confirm"],
        "keltner_adx_dmi": [
            "keltner_long_confirm",
            "keltner_short_confirm",
            "dmi_long_confirm",
            "dmi_short_confirm",
        ],
        "keltner_adx_dmi_vwap": [
            "keltner_long_confirm",
            "keltner_short_confirm",
            "dmi_long_confirm",
            "dmi_short_confirm",
            "vwap_long_confirm",
            "vwap_short_confirm",
        ],
        "keltner_adx_dmi_vwap_structure": [
            "keltner_long_confirm",
            "keltner_short_confirm",
            "dmi_long_confirm",
            "dmi_short_confirm",
            "vwap_long_confirm",
            "vwap_short_confirm",
            "structure_long_confirm",
            "structure_short_confirm",
        ],
    }
    if battery not in required:
        raise ValueError(f"Unknown graphical battery: {battery}")

    for column in required[battery]:
        if column not in c.columns:
            raise ValueError(f"Missing graphical feature: {column}")

    long_ok = base.eq("LONG") & c["keltner_long_confirm"]
    short_ok = base.eq("SHORT") & c["keltner_short_confirm"]

    if battery in {"keltner_adx_dmi", "keltner_adx_dmi_vwap", "keltner_adx_dmi_vwap_structure"}:
        long_ok &= c["dmi_long_confirm"]
        short_ok &= c["dmi_short_confirm"]

    if battery in {"keltner_adx_dmi_vwap", "keltner_adx_dmi_vwap_structure"}:
        long_ok &= c["vwap_long_confirm"]
        short_ok &= c["vwap_short_confirm"]

    if battery == "keltner_adx_dmi_vwap_structure":
        long_ok &= c["structure_long_confirm"]
        short_ok &= c["structure_short_confirm"]

    c["decision"] = "NO_TRADE"
    c.loc[long_ok, "decision"] = "LONG"
    c.loc[short_ok, "decision"] = "SHORT"
    return c


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

    engine = OperationalBacktest(
        fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")),
        slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")),
    )

    out = Path("artifacts/graphical-benchmark")
    out.mkdir(parents=True, exist_ok=True)

    configs: list[tuple[str, pd.DataFrame]] = [
        ("frozen_core_relative_funding_cvd_premium", core),
        ("battery_1_keltner", apply_graphical_battery(core, "keltner")),
        ("battery_2_keltner_adx_dmi", apply_graphical_battery(core, "keltner_adx_dmi")),
        ("battery_3_keltner_adx_dmi_vwap", apply_graphical_battery(core, "keltner_adx_dmi_vwap")),
        (
            "battery_4_keltner_adx_dmi_vwap_structure",
            apply_graphical_battery(core, "keltner_adx_dmi_vwap_structure"),
        ),
    ]

    rows = []
    for mode, decisions in configs:
        for rr in (1.0, 1.5, 2.0, 3.0):
            trades = engine.run(decisions, decisions[["timestamp", "decision"]], rr=rr)
            met = engine.metrics(trades)
            met.update({"mode": mode, "rr": rr})
            rows.append(met)
            trades.to_csv(out / f"trades_{mode}_rr_{rr}.csv", index=False)

    results = pd.DataFrame(rows)[
        [
            "mode",
            "rr",
            "trades",
            "win_rate_pct",
            "net_return_pct",
            "profit_factor",
            "expectancy_pct",
            "max_drawdown_pct",
            "payoff",
            "long",
            "short",
        ]
    ]
    results.to_csv(out / "results.csv", index=False)

    cfg = graphical.config
    manifest = {
        "benchmark": "graphical-entry-batteries-v1",
        "base": "frozen core = Binance OI > 0 + relative funding-z + CVD + relative premium-z",
        "symbol": symbol,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "signal_timeframe": "15m",
        "entry_execution": "signal at t, entry at t+1 open",
        "exit_control": "unchanged fixed ATR stop and R-multiple target",
        "keltner": {
            "ema_period": cfg.keltner_ema_period,
            "atr_period": cfg.atr_period,
            "atr_multiplier": cfg.keltner_atr_multiplier,
            "confirmation": "trend-side of middle band AND channel width expanding",
        },
        "adx_dmi": {
            "period": cfg.adx_period,
            "confirmation": "directional DI agrees AND ADX rising; no fitted ADX threshold",
        },
        "vwap": {
            "anchor": "00:00 UTC daily session",
            "confirmation": "LONG close > VWAP; SHORT close < VWAP",
        },
        "market_structure": {
            "lookback_bars": cfg.market_structure_lookback,
            "confirmation": "close breaks prior 20-bar high/low; current bar excluded",
        },
        "fee_bps_per_side": engine.fee_bps_per_side,
        "slippage_bps_per_side": engine.slippage_bps_per_side,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
