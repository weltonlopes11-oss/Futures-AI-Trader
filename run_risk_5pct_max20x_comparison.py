from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig
from run_regime_discovery_2025_validate_2026 import run_featured_regime_window

ROBUST_REGIMES = {
    "LONG|LOW|CHOP",
    "SHORT|NORMAL|MIXED",
}

INITIAL_EQUITY_BRL = 2000.0


def simulate(trades: pd.DataFrame, risk_fraction: float, max_leverage: float) -> tuple[pd.DataFrame, dict]:
    out = trades.copy().sort_values("entry_time").reset_index(drop=True)
    equity = INITIAL_EQUITY_BRL
    peak = equity
    max_dd_pct = 0.0
    max_dd_brl = 0.0
    max_loss = 0.0
    max_gain = 0.0
    leverages = []
    rows = []

    for _, row in out.iterrows():
        entry = float(row["entry_price"])
        stop = float(row["stop_price"])
        stop_distance = abs(entry - stop) / entry
        if not np.isfinite(stop_distance) or stop_distance <= 0:
            raise ValueError("Invalid stop distance")

        leverage = min(max_leverage, risk_fraction / stop_distance)
        notional = equity * leverage
        pnl = notional * (float(row["net_return_pct"]) / 100.0)
        before = equity
        equity = before + pnl

        peak = max(peak, equity)
        dd_brl = equity - peak
        dd_pct = (equity / peak - 1.0) * 100.0 if peak > 0 else float("nan")
        max_dd_brl = min(max_dd_brl, dd_brl)
        max_dd_pct = min(max_dd_pct, dd_pct)
        max_loss = min(max_loss, pnl)
        max_gain = max(max_gain, pnl)
        leverages.append(leverage)

        r = row.to_dict()
        r.update({
            "equity_before_brl": before,
            "risk_budget_brl": before * risk_fraction,
            "stop_distance_pct": stop_distance * 100.0,
            "sizing_leverage": leverage,
            "notional_brl": notional,
            "trade_pnl_brl": pnl,
            "equity_after_brl": equity,
            "drawdown_pct_after_trade": dd_pct,
        })
        rows.append(r)

    sized = pd.DataFrame(rows)
    summary = {
        "risk_fraction": risk_fraction,
        "max_leverage": max_leverage,
        "initial_equity_brl": INITIAL_EQUITY_BRL,
        "ending_equity_brl": float(equity),
        "total_pnl_brl": float(equity - INITIAL_EQUITY_BRL),
        "return_pct": float((equity / INITIAL_EQUITY_BRL - 1.0) * 100.0),
        "max_drawdown_pct": float(max_dd_pct),
        "max_drawdown_brl": float(max_dd_brl),
        "min_leverage_used": float(min(leverages)),
        "avg_leverage_used": float(np.mean(leverages)),
        "max_leverage_used": float(max(leverages)),
        "trades_at_max_leverage": int(sum(abs(x - max_leverage) < 1e-12 for x in leverages)),
        "max_single_loss_brl": float(max_loss),
        "max_single_gain_brl": float(max_gain),
    }
    return sized, summary


def main() -> None:
    cfg = IchimokuKeltner1HConfig(
        fee_bps_per_side=4.0,
        slippage_bps_per_side=1.0,
    )

    eval_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    eval_end = datetime(2026, 9, 1, tzinfo=timezone.utc)

    loader = BinanceDataVisionLoader()
    warmup = eval_start - timedelta(days=120)
    h1 = loader.fetch_window("ETHUSDT", "1h", warmup, eval_end)
    h4 = loader.fetch_window("ETHUSDT", "4h", warmup - timedelta(days=15), eval_end)

    trades, _ = run_featured_regime_window(
        h1,
        h4,
        eval_start,
        eval_end,
        cfg,
        allowed_regimes=ROBUST_REGIMES,
    )

    baseline_trades, baseline = simulate(trades, risk_fraction=0.10, max_leverage=15.0)
    candidate_trades, candidate = simulate(trades, risk_fraction=0.05, max_leverage=20.0)

    comparison = {
        "experiment_id": "risk-5pct-max20x-vs-risk10pct-max15x",
        "period": "2024-01-01 through 2026-08-31 UTC",
        "rules_unchanged_except_sizing": True,
        "trades": int(len(trades)),
        "baseline_10pct_15x": baseline,
        "candidate_5pct_20x": candidate,
        "differences_candidate_minus_baseline": {
            "ending_equity_brl": candidate["ending_equity_brl"] - baseline["ending_equity_brl"],
            "return_pct_points": candidate["return_pct"] - baseline["return_pct"],
            "max_drawdown_pct_points": candidate["max_drawdown_pct"] - baseline["max_drawdown_pct"],
            "avg_leverage": candidate["avg_leverage_used"] - baseline["avg_leverage_used"],
        },
    }

    out = Path("artifacts/risk-5pct-max20x-2024-2026")
    out.mkdir(parents=True, exist_ok=True)
    baseline_trades.to_csv(out / "trades_baseline_10pct_15x.csv", index=False)
    candidate_trades.to_csv(out / "trades_candidate_5pct_20x.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print(json.dumps(comparison, indent=2, default=str))


if __name__ == "__main__":
    main()
