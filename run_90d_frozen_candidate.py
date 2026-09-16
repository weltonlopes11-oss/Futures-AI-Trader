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
from run_graphical_benchmark import apply_graphical_battery
from run_operational_benchmark import atr, build_decisions, causal_context, causal_event_zscore, causal_relative_zscore, require_coverage, trend


def longest_losing_streak(trades: pd.DataFrame) -> int:
    best = cur = 0
    for x in trades.get("net_return_pct", pd.Series(dtype=float)):
        if float(x) < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main():
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    eval_start = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_START_UTC", "2026-06-17T00:00:00+00:00").replace("Z", "+00:00"))
    eval_end = datetime.fromisoformat(os.getenv("BACKTEST_EVAL_END_UTC", "2026-09-15T00:00:00+00:00").replace("Z", "+00:00"))
    if eval_start.tzinfo is None: eval_start = eval_start.replace(tzinfo=timezone.utc)
    if eval_end.tzinfo is None: eval_end = eval_end.replace(tzinfo=timezone.utc)
    warmup_days = int(os.getenv("BACKTEST_WARMUP_DAYS", "14"))
    fetch_start = eval_start - timedelta(days=warmup_days)

    prices = BinanceDataVisionLoader()
    metrics = BinanceMetricsDataVisionLoader()
    funding_loader = BinanceFundingRateLoader()
    funding_loader.BASE_URL = os.getenv("BACKTEST_FUNDING_URL", "https://api-dev.pipai.org/fapi/v1/fundingRate")
    positioning_loader = BinancePositioningContextLoader()
    cvd_builder = CumulativeVolumeDelta(fast=12, slow=26)
    graphical = GraphicalContext(GraphicalContextConfig())

    m15 = prices.fetch_window(symbol, "15m", fetch_start, eval_end)
    h1 = prices.fetch_window(symbol, "1h", fetch_start, eval_end)
    h4 = prices.fetch_window(symbol, "4h", fetch_start, eval_end)
    oi = metrics.fetch_window(symbol, fetch_start, eval_end)
    funding = funding_loader.fetch_window(symbol, fetch_start, eval_end, snapshot_path=None)
    funding["funding_z"] = causal_event_zscore(funding["funding_rate"])
    premium = positioning_loader.fetch_premium_index(symbol, fetch_start, eval_end, interval="15m")

    mins = int((eval_end - fetch_start).total_seconds() // 60)
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

    core = build_decisions(context, oi_mode="binance", funding_mode="relative", use_cvd=True, use_premium=True, use_positioning=False, use_leverage_stress=False)
    keltner = apply_graphical_battery(core, "keltner")
    anti = AntiExtensionFilter(AntiExtensionConfig(max_vwap_distance_atr=2.0, max_keltner_position=0.75))
    decisions = anti.apply(keltner, "keltner_only")

    eval_start_naive = eval_start.astimezone(timezone.utc).replace(tzinfo=None)
    eval_end_naive = eval_end.astimezone(timezone.utc).replace(tzinfo=None)
    decisions["timestamp"] = pd.to_datetime(decisions["timestamp"], errors="coerce")
    decisions = decisions[(decisions["timestamp"] >= eval_start_naive) & (decisions["timestamp"] < eval_end_naive)].reset_index(drop=True)

    funding_cov = float(decisions["funding_z"].notna().mean() * 100.0)
    premium_cov = float(decisions["premium_z"].notna().mean() * 100.0)
    oi_cov = float(decisions["open_interest_change_pct"].notna().mean() * 100.0)

    engine = StructuralExitBacktest(
        fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")),
        slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")),
        structure_lookback=5,
    )
    trades = engine.run(decisions, decisions[["timestamp", "decision"]], rr=2.0)
    met = engine.metrics(trades)
    met["longest_losing_streak"] = longest_losing_streak(trades)

    out = Path("artifacts/frozen-candidate-90d")
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    pd.DataFrame([met]).to_csv(out / "metrics.csv", index=False)

    manifest = {
        "benchmark": "frozen-candidate-90d-v1",
        "strategy": "frozen core + Keltner confirmation + Keltner anti-extension 0.75 + 1ATR stop + 2R target + 5-bar structural exit",
        "symbol": symbol,
        "evaluation_start_utc": eval_start.isoformat(),
        "evaluation_end_utc": eval_end.isoformat(),
        "warmup_days": warmup_days,
        "fetch_start_utc": fetch_start.isoformat(),
        "funding_transport": funding_loader.BASE_URL,
        "funding_semantics": "Binance USD-M /fapi/v1/fundingRate passthrough; strategy definition unchanged",
        "fee_bps_per_side": engine.fee_bps_per_side,
        "slippage_bps_per_side": engine.slippage_bps_per_side,
        "coverage_pct": {"funding_z": funding_cov, "premium_z": premium_cov, "open_interest_change": oi_cov},
        "anti_overfit": "No strategy parameter was changed for this 90-day validation window.",
        "metrics": met,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    if len(trades): print("\nTRADES\n" + trades.to_string(index=False))


if __name__ == "__main__":
    main()
