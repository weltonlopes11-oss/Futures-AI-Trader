from __future__ import annotations

import os
from pathlib import Path

from backtest.engine import BacktestEngine
from backtest.fixed_historical_loader import FixedHistoricalDataLoader
from backtest.oi_profit_analyzer import OIProfitAnalyzer
from filters.regime_filter import RegimeFilter
from filters.timeframe_filter import TimeframeFilter
from filters.trend_filter import TrendFilter
from filters.volatility_filter import VolatilityFilter


DATA_PATH = os.getenv(
    "BACKTEST_FIXED_DATA_PATH",
    str(Path("data") / "ETHUSDT_1m_baseline_oi.csv"),
)
SYMBOL = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
INTERVAL = os.getenv("BACKTEST_INTERVAL", "1m")
LIMIT = int(os.getenv("BACKTEST_LIMIT", "20000"))
WARMUP = int(os.getenv("BACKTEST_WARMUP", "200"))
FEE_BPS_PER_SIDE = float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "0"))
SLIPPAGE_BPS_PER_SIDE = float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "0"))


def build_filters():
    return [
        RegimeFilter(),
        TrendFilter(),
        VolatilityFilter(),
        TimeframeFilter(),
    ]


def print_metrics(metrics):
    return (
        f"n={metrics['count']:<4} "
        f"net_avg={metrics['avg_net_return_pct']:>9.5f}% "
        f"net_med={metrics['median_net_return_pct']:>9.5f}% "
        f"net_win={metrics['net_positive_rate_pct']:>6.2f}% "
        f"MFE={metrics['avg_mfe_pct']:>8.5f}% "
        f"MAE={metrics['avg_mae_pct']:>8.5f}% "
        f"MFE/MAE={metrics['mfe_mae_ratio']:>6.3f}"
    )


def main():
    print()
    print("=" * 78)
    print("OI PROFIT / MFE / MAE ANALYSIS")
    print("=" * 78)
    print(f"Dataset.................: {DATA_PATH}")
    print(f"Fee por lado............: {FEE_BPS_PER_SIDE:.3f} bps")
    print(f"Slippage por lado.......: {SLIPPAGE_BPS_PER_SIDE:.3f} bps")

    if FEE_BPS_PER_SIDE == 0 and SLIPPAGE_BPS_PER_SIDE == 0:
        print("ATENÇÃO.................: custos zerados; net_return equivale ao retorno bruto")

    candles = FixedHistoricalDataLoader(DATA_PATH).load(LIMIT)

    engine = BacktestEngine(
        filters=build_filters(),
        risk_validators=[],
        symbol=SYMBOL,
        interval=INTERVAL,
        data_source="FIXED",
        fixed_data_path=DATA_PATH,
    )
    engine.run(limit=LIMIT, warmup=WARMUP)

    analyzer = OIProfitAnalyzer(
        records=engine.telemetry_storage.records,
        candles=candles,
        horizons=(5, 15, 30, 60),
        fee_bps_per_side=FEE_BPS_PER_SIDE,
        slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
    )

    print()
    print("OI IMPULSE THRESHOLDS")
    print(f"P10.....................: {analyzer.thresholds['p10']:.6f}%")
    print(f"P25.....................: {analyzer.thresholds['p25']:.6f}%")
    print(f"P75.....................: {analyzer.thresholds['p75']:.6f}%")
    print(f"P90.....................: {analyzer.thresholds['p90']:.6f}%")
    print(f"Custo round-trip........: {analyzer.round_trip_cost_pct:.6f}%")

    summary = analyzer.summary_by_decision_and_bucket()

    print()
    print("NET RETURN BY DECISION / OI IMPULSE")

    for horizon in sorted(summary):
        print()
        print(f"--- HORIZON {horizon}m ---")
        for decision in ("LONG", "SHORT"):
            buckets = summary[horizon].get(decision, {})
            if not buckets:
                continue
            print(decision)
            for bucket in (
                "STRONG_DROP",
                "DROP",
                "NORMAL",
                "RISE",
                "STRONG_RISE",
            ):
                metrics = buckets.get(bucket)
                if metrics:
                    print(f"  {bucket:<14} {print_metrics(metrics)}")

    candidates = []
    for horizon, decisions in summary.items():
        for decision, buckets in decisions.items():
            if decision not in {"LONG", "SHORT"}:
                continue
            for bucket, metrics in buckets.items():
                if metrics["count"] < 30:
                    continue
                candidates.append(
                    (
                        metrics["avg_net_return_pct"],
                        metrics["mfe_mae_ratio"],
                        metrics["count"],
                        horizon,
                        decision,
                        bucket,
                        metrics,
                    )
                )

    print()
    print("TOP CANDIDATES BY AVERAGE NET RETURN (min n=30)")

    for _, _, _, horizon, decision, bucket, metrics in sorted(
        candidates,
        reverse=True,
    )[:15]:
        print(
            f"{horizon:>2}m | {decision:<5} | {bucket:<14} | "
            f"{print_metrics(metrics)}"
        )

    print()
    print("=" * 78)
    print("Analysis only: no trading rule was changed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
