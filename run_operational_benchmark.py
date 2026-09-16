from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.binance_funding_rate import BinanceFundingRateLoader
from backtest.binance_metrics_data_vision import BinanceMetricsDataVisionLoader
from backtest.binance_positioning_context import BinancePositioningContextLoader
from backtest.bybit_derivatives_history import BybitDerivativesHistoryLoader
from backtest.crypto_leverage_stress import CryptoLeverageStressScore
from backtest.cvd import CumulativeVolumeDelta
from backtest.operational_backtest import OperationalBacktest


def atr(frame, period=14):
    prev = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev).abs(),
            (frame["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def trend(frame, fast=12, slow=26):
    f = frame["close"].ewm(span=fast, adjust=False).mean()
    s = frame["close"].ewm(span=slow, adjust=False).mean()
    return pd.Series(pd.NA, index=frame.index).mask(f > s, "LONG").mask(f < s, "SHORT")


def causal_context(signal, higher, label):
    h = higher.copy()
    h[label] = trend(h)
    h["available_at"] = pd.to_datetime(h["close_time"], errors="coerce")
    return pd.merge_asof(
        signal.sort_values("timestamp"),
        h[["available_at", label]].dropna().sort_values("available_at"),
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    )


def require_coverage(frame, expected, label):
    if len(frame) < int(expected * 0.99):
        raise RuntimeError(f"Insufficient {label} coverage: {len(frame)}/{expected}")


def positioning_pressure(frame: pd.DataFrame) -> pd.Series:
    cols = ["global_ls_ratio", "top_account_ls_ratio", "top_position_ls_ratio"]
    available = [c for c in cols if c in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index)
    logs = pd.concat(
        [np.log(pd.to_numeric(frame[c], errors="coerce").clip(lower=1e-9)) for c in available],
        axis=1,
    )
    return logs.mean(axis=1)


def causal_relative_zscore(
    series: pd.Series,
    window: int = 96,
    min_periods: int = 48,
) -> pd.Series:
    """Current observation versus prior history only; no look-ahead."""
    s = pd.to_numeric(series, errors="coerce")
    prior = s.shift(1)
    mean = prior.rolling(window=window, min_periods=min_periods).mean()
    std = prior.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return ((s - mean) / std.replace(0.0, np.nan)).clip(-5.0, 5.0)


def causal_event_zscore(
    series: pd.Series,
    window: int = 21,
    min_periods: int = 6,
) -> pd.Series:
    """Causal z-score for sparse funding events.

    Funding settles much less frequently than 15m bars. Scoring the event series
    before candle alignment avoids counting the same settled funding value dozens
    of times. Twenty-one prior events is roughly one week at an 8h cadence; six
    events provide a two-day warm-up. Both are fixed calendar choices, not fitted
    to trade PnL.
    """
    return causal_relative_zscore(series, window=window, min_periods=min_periods)


def strict_mean(columns: list[pd.Series]) -> pd.Series:
    frame = pd.concat(columns, axis=1)
    valid = frame.notna().all(axis=1)
    return frame.mean(axis=1).where(valid)


def build_decisions(
    context,
    oi_mode="none",
    funding_mode="none",
    use_cvd=False,
    use_premium=False,
    use_positioning=False,
    use_leverage_stress=False,
):
    c = context.copy()
    c["decision"] = "NO_TRADE"

    base_long = (
        (c.signal_15m == "LONG")
        & (c.trend_1h == "LONG")
        & (c.trend_4h == "LONG")
    )
    base_short = (
        (c.signal_15m == "SHORT")
        & (c.trend_1h == "SHORT")
        & (c.trend_4h == "SHORT")
    )

    if oi_mode == "binance":
        oi_confirm = c.open_interest_change_pct > 0
    elif oi_mode == "cross":
        oi_confirm = c.cross_oi_change_pct > 0
    elif oi_mode == "none":
        oi_confirm = pd.Series(True, index=c.index)
    else:
        raise ValueError(f"Unknown oi_mode: {oi_mode}")

    if funding_mode == "absolute":
        funding_long = c.funding_rate <= 0
        funding_short = c.funding_rate >= 0
    elif funding_mode == "relative":
        funding_long = c.funding_z <= 0
        funding_short = c.funding_z >= 0
    elif funding_mode == "cross":
        funding_long = c.cross_funding_z <= 0
        funding_short = c.cross_funding_z >= 0
    elif funding_mode == "none":
        funding_long = pd.Series(True, index=c.index)
        funding_short = pd.Series(True, index=c.index)
    else:
        raise ValueError(f"Unknown funding_mode: {funding_mode}")

    if use_cvd:
        cvd_long = c.cvd_signal == "LONG"
        cvd_short = c.cvd_signal == "SHORT"
    else:
        cvd_long = pd.Series(True, index=c.index)
        cvd_short = pd.Series(True, index=c.index)

    if use_premium:
        premium_long = c.premium_z <= 0
        premium_short = c.premium_z >= 0
    else:
        premium_long = pd.Series(True, index=c.index)
        premium_short = pd.Series(True, index=c.index)

    if use_positioning:
        positioning_long = c.positioning_z <= 0
        positioning_short = c.positioning_z >= 0
    else:
        positioning_long = pd.Series(True, index=c.index)
        positioning_short = pd.Series(True, index=c.index)

    if use_leverage_stress:
        stress_long = c.crypto_leverage_stress_score <= 0
        stress_short = c.crypto_leverage_stress_score >= 0
    else:
        stress_long = pd.Series(True, index=c.index)
        stress_short = pd.Series(True, index=c.index)

    long_ok = (
        base_long
        & oi_confirm
        & funding_long
        & cvd_long
        & premium_long
        & positioning_long
        & stress_long
    )
    short_ok = (
        base_short
        & oi_confirm
        & funding_short
        & cvd_short
        & premium_short
        & positioning_short
        & stress_short
    )
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
    bybit_loader = BybitDerivativesHistoryLoader()
    cvd_builder = CumulativeVolumeDelta(fast=12, slow=26)
    leverage_builder = CryptoLeverageStressScore()

    funding_snapshot = Path("research_snapshots/ETHUSDT_funding_2026-09-01_2026-09-15.csv")

    m15 = prices.fetch_window(symbol, "15m", start, end)
    h1 = prices.fetch_window(symbol, "1h", start, end)
    h4 = prices.fetch_window(symbol, "4h", start, end)
    oi = metrics.fetch_window(symbol, start, end)
    funding = funding_loader.fetch_window(symbol, start, end, snapshot_path=funding_snapshot)
    funding["funding_z"] = causal_event_zscore(funding["funding_rate"])
    premium = positioning_loader.fetch_premium_index(symbol, start, end, interval="15m")

    bybit_oi = bybit_loader.fetch_open_interest(symbol, start, end, interval="15min")
    bybit_funding = bybit_loader.fetch_funding(symbol, start, end)
    bybit_funding["bybit_funding_z"] = causal_event_zscore(
        bybit_funding["bybit_funding_rate"]
    )

    mins = int((end - start).total_seconds() // 60)
    require_coverage(m15, mins // 15, "15m")
    require_coverage(h1, mins // 60, "1h")
    require_coverage(h4, mins // 240, "4h")
    require_coverage(bybit_oi, mins // 15, "Bybit 15m OI")

    m15["atr"] = atr(m15)
    m15["signal_15m"] = trend(m15)
    m15 = cvd_builder.enrich(m15)

    context = causal_context(m15, h1, "trend_1h")
    context = causal_context(context, h4, "trend_4h")
    context = metrics.align_causally(context, oi)
    context = funding_loader.align_causally(context, funding)
    context = positioning_loader.align_causally(context, premium)
    context = bybit_loader.align_oi_causally(context, bybit_oi)
    context = bybit_loader.align_funding_causally(context, bybit_funding)

    context["positioning_pressure"] = positioning_pressure(context)
    context["premium_z"] = causal_relative_zscore(context["premium_close"])
    context["positioning_z"] = causal_relative_zscore(context["positioning_pressure"])
    context["cross_oi_change_pct"] = strict_mean(
        [context["open_interest_change_pct"], context["bybit_open_interest_change_pct"]]
    )
    context["cross_funding_z"] = strict_mean(
        [context["funding_z"], context["bybit_funding_z"]]
    )
    context = leverage_builder.enrich(context)

    oi_coverage = context.open_interest.notna().mean() * 100
    funding_coverage = context.funding_rate.notna().mean() * 100
    funding_z_coverage = context.funding_z.notna().mean() * 100
    cvd_coverage = context.cvd_signal.notna().mean() * 100
    premium_coverage = context.premium_close.notna().mean() * 100
    positioning_coverage = context.positioning_pressure.notna().mean() * 100
    premium_z_coverage = context.premium_z.notna().mean() * 100
    positioning_z_coverage = context.positioning_z.notna().mean() * 100
    bybit_oi_coverage = context.bybit_open_interest.notna().mean() * 100
    bybit_funding_coverage = context.bybit_funding_rate.notna().mean() * 100
    bybit_funding_z_coverage = context.bybit_funding_z.notna().mean() * 100
    cross_oi_coverage = context.cross_oi_change_pct.notna().mean() * 100
    cross_funding_coverage = context.cross_funding_z.notna().mean() * 100
    stress_coverage = context.crypto_leverage_stress_score.notna().mean() * 100

    if oi_coverage < 99:
        raise RuntimeError(f"Insufficient OI coverage: {oi_coverage:.2f}%")
    if funding_coverage < 99:
        raise RuntimeError(f"Insufficient funding coverage: {funding_coverage:.2f}%")
    if funding_z_coverage < 75:
        raise RuntimeError(f"Insufficient relative funding coverage: {funding_z_coverage:.2f}%")
    if cvd_coverage < 99:
        raise RuntimeError(f"Insufficient CVD coverage: {cvd_coverage:.2f}%")
    if premium_coverage < 95:
        raise RuntimeError(f"Insufficient premium coverage: {premium_coverage:.2f}%")
    if positioning_coverage < 95:
        raise RuntimeError(f"Insufficient positioning coverage: {positioning_coverage:.2f}%")
    if premium_z_coverage < 90:
        raise RuntimeError(f"Insufficient causal premium-z coverage: {premium_z_coverage:.2f}%")
    if positioning_z_coverage < 90:
        raise RuntimeError(f"Insufficient causal positioning-z coverage: {positioning_z_coverage:.2f}%")
    if bybit_oi_coverage < 99:
        raise RuntimeError(f"Insufficient Bybit OI coverage: {bybit_oi_coverage:.2f}%")
    if bybit_funding_coverage < 95:
        raise RuntimeError(f"Insufficient Bybit funding coverage: {bybit_funding_coverage:.2f}%")
    if bybit_funding_z_coverage < 75:
        raise RuntimeError(
            f"Insufficient Bybit relative funding coverage: {bybit_funding_z_coverage:.2f}%"
        )
    if cross_oi_coverage < 99:
        raise RuntimeError(f"Insufficient cross-exchange OI coverage: {cross_oi_coverage:.2f}%")
    if cross_funding_coverage < 75:
        raise RuntimeError(
            f"Insufficient cross-exchange funding coverage: {cross_funding_coverage:.2f}%"
        )

    engine = OperationalBacktest(
        fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")),
        slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")),
    )

    out = Path("artifacts/operational-benchmark")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    modes = (
        ("baseline", "none", "none", False, False, False, False),
        ("open_interest", "binance", "none", False, False, False, False),
        ("oi_funding_absolute", "binance", "absolute", False, False, False, False),
        ("oi_funding_absolute_cvd", "binance", "absolute", True, False, False, False),
        ("oi_funding_absolute_cvd_premium", "binance", "absolute", True, True, False, False),
        ("oi_funding_absolute_cvd_premium_positioning", "binance", "absolute", True, True, True, False),
        ("oi_funding_absolute_cvd_premium_positioning_stress", "binance", "absolute", True, True, True, True),
        ("oi_funding_relative", "binance", "relative", False, False, False, False),
        ("oi_funding_relative_cvd", "binance", "relative", True, False, False, False),
        ("oi_funding_relative_cvd_premium", "binance", "relative", True, True, False, False),
        ("cross_oi_funding_relative", "cross", "cross", False, False, False, False),
        ("cross_oi_funding_relative_cvd", "cross", "cross", True, False, False, False),
        ("cross_oi_funding_relative_cvd_premium", "cross", "cross", True, True, False, False),
    )

    for mode, oi_mode, funding_mode, use_cvd, use_premium, use_positioning, use_stress in modes:
        decisions = build_decisions(
            context,
            oi_mode=oi_mode,
            funding_mode=funding_mode,
            use_cvd=use_cvd,
            use_premium=use_premium,
            use_positioning=use_positioning,
            use_leverage_stress=use_stress,
        )
        for rr in (1.0, 1.5, 2.0, 3.0):
            trades = engine.run(decisions, decisions[["timestamp", "decision"]], rr=rr)
            met = engine.metrics(trades)
            met.update({"mode": mode, "rr": rr})
            rows.append(met)
            trades.to_csv(out / f"trades_{mode}_rr_{rr}.csv", index=False)

    results = pd.DataFrame(rows)[
        [
            "mode", "rr", "trades", "win_rate_pct", "net_return_pct",
            "profit_factor", "expectancy_pct", "max_drawdown_pct", "payoff",
            "long", "short",
        ]
    ]
    results.to_csv(out / "results.csv", index=False)

    manifest = {
        "source": "Binance Data Vision/funding snapshot + Bybit V5 public derivatives history",
        "market": "ETH USDT linear perpetuals",
        "symbol": symbol,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "signal": "15m",
        "structure": "1h",
        "regime": "4h",
        "open_interest": "Binance metrics + Bybit V5 15m OI, causal backward alignment",
        "oi_rule_absolute_control": "Binance open_interest_change_pct > 0",
        "cross_oi_rule": "mean(Binance dOI%, Bybit dOI%) > 0; both venues required",
        "oi_observations": len(oi),
        "bybit_oi_observations": len(bybit_oi),
        "oi_coverage_pct": oi_coverage,
        "bybit_oi_coverage_pct": bybit_oi_coverage,
        "cross_oi_coverage_pct": cross_oi_coverage,
        "funding_absolute_control": "LONG Binance funding <= 0; SHORT >= 0",
        "funding_relative_rule": "Binance funding event z-score: LONG <= 0; SHORT >= 0",
        "cross_funding_rule": "mean(Binance funding-z, Bybit funding-z): LONG <= 0; SHORT >= 0; both venues required",
        "funding_event_z_window": 21,
        "funding_event_z_min_periods": 6,
        "funding_coverage_pct": funding_coverage,
        "funding_z_coverage_pct": funding_z_coverage,
        "bybit_funding_observations": len(bybit_funding),
        "bybit_funding_coverage_pct": bybit_funding_coverage,
        "bybit_funding_z_coverage_pct": bybit_funding_z_coverage,
        "cross_funding_coverage_pct": cross_funding_coverage,
        "cvd_rule": "LONG CVD EMA12 > EMA26; SHORT CVD EMA12 < EMA26",
        "cvd_coverage_pct": cvd_coverage,
        "premium_rule": "closed 15m premium; LONG causal premium z <= 0; SHORT >= 0; 96-bar prior window",
        "premium_coverage_pct": premium_coverage,
        "premium_z_coverage_pct": premium_z_coverage,
        "positioning_rule": "log-ratio pressure vs causal 96-bar prior distribution; LONG z <= 0; SHORT z >= 0",
        "positioning_coverage_pct": positioning_coverage,
        "positioning_z_coverage_pct": positioning_z_coverage,
        "leverage_stress_rule": "LONG stress <= 0; SHORT stress >= 0; sign only, no tuned threshold",
        "leverage_stress_coverage_pct": stress_coverage,
        "fee_bps_per_side": engine.fee_bps_per_side,
        "slippage_bps_per_side": engine.slippage_bps_per_side,
        "candles_15m": len(m15),
        "candles_1h": len(h1),
        "candles_4h": len(h4),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
