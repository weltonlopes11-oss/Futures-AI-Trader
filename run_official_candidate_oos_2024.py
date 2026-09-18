from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, metrics
from run_regime_discovery_2025_validate_2026 import run_featured_regime_window

ROBUST_REGIMES = {
    "LONG|LOW|CHOP",
    "SHORT|NORMAL|MIXED",
}

BASE_BRL = 2000.0
MAX_LEVERAGE = 15.0
RISK_PER_TRADE_BRL = 200.0


def add_dynamic_sizing(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = trades.copy()
    if out.empty:
        return out, {
            "base_brl": BASE_BRL,
            "max_leverage": MAX_LEVERAGE,
            "risk_per_trade_brl": RISK_PER_TRADE_BRL,
            "total_pnl_brl": 0.0,
            "ending_equity_brl": BASE_BRL,
            "max_drawdown_brl": 0.0,
            "max_drawdown_pct_of_start": 0.0,
        }

    stop_distance_pct = (
        (out["entry_price"].astype(float) - out["stop_price"].astype(float)).abs()
        / out["entry_price"].astype(float)
    )
    risk_fraction = RISK_PER_TRADE_BRL / BASE_BRL
    leverage = (risk_fraction / stop_distance_pct).clip(upper=MAX_LEVERAGE)

    out["stop_distance_pct"] = stop_distance_pct * 100.0
    out["sizing_leverage"] = leverage
    out["notional_brl"] = BASE_BRL * leverage
    out["trade_pnl_brl"] = out["notional_brl"] * (out["net_return_pct"].astype(float) / 100.0)
    out["equity_brl"] = BASE_BRL + out["trade_pnl_brl"].cumsum()

    equity_curve = pd.concat(
        [pd.Series([BASE_BRL], dtype=float), out["equity_brl"].reset_index(drop=True)],
        ignore_index=True,
    )
    running_peak = equity_curve.cummax()
    dd_brl = equity_curve - running_peak
    dd_pct_start = dd_brl / BASE_BRL * 100.0

    summary = {
        "base_brl": BASE_BRL,
        "max_leverage": MAX_LEVERAGE,
        "risk_per_trade_brl": RISK_PER_TRADE_BRL,
        "trades": int(len(out)),
        "min_leverage": float(out["sizing_leverage"].min()),
        "avg_leverage": float(out["sizing_leverage"].mean()),
        "max_leverage_used": float(out["sizing_leverage"].max()),
        "total_pnl_brl": float(out["trade_pnl_brl"].sum()),
        "ending_equity_brl": float(out["equity_brl"].iloc[-1]),
        "max_drawdown_brl": float(dd_brl.min()),
        "max_drawdown_pct_of_start": float(dd_pct_start.min()),
        "max_single_loss_brl": float(out["trade_pnl_brl"].min()),
        "max_single_gain_brl": float(out["trade_pnl_brl"].max()),
    }
    return out, summary


def main() -> None:
    cfg = IchimokuKeltner1HConfig(
        fee_bps_per_side=4.0,
        slippage_bps_per_side=1.0,
    )

    eval_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    eval_end = datetime(2025, 1, 1, tzinfo=timezone.utc)

    loader = BinanceDataVisionLoader()
    warmup = eval_start - timedelta(days=120)
    h1 = loader.fetch_window("ETHUSDT", "1h", warmup, eval_end)
    h4 = loader.fetch_window("ETHUSDT", "4h", warmup - timedelta(days=15), eval_end)

    trades, counts = run_featured_regime_window(
        h1,
        h4,
        eval_start,
        eval_end,
        cfg,
        allowed_regimes=ROBUST_REGIMES,
    )

    sized, sizing_summary = add_dynamic_sizing(trades)

    result = {
        "experiment_id": "official-candidate-oos-2024",
        "status": "out_of_sample_validation_not_promoted",
        "official_strategy_modified": False,
        "period": "2024-01-01 through 2024-12-31 UTC",
        "rules_frozen_before_test": True,
        "rules": {
            "symbol": "ETHUSDT",
            "entry_timeframe": "1h",
            "direction_timeframe": "4h",
            "allowed_regimes": sorted(ROBUST_REGIMES),
            "volatility_definition": "ATR20/close vs trailing 90d percentiles: LOW <= q33, NORMAL q33-q67, HIGH > q67",
            "efficiency_definition": "ER20: CHOP <0.15, MIXED 0.15-<0.30, EFFICIENT >=0.30",
            "stop": "2x ATR20",
            "target": "Keltner executable band",
            "fees_bps_per_side": 4.0,
            "slippage_bps_per_side": 1.0,
            "base_brl": BASE_BRL,
            "risk_per_trade_brl": RISK_PER_TRADE_BRL,
            "max_leverage": MAX_LEVERAGE,
            "sizing_formula": "leverage = min(15x, (200/2000) / stop_distance_fraction)",
        },
        "trade_metrics": metrics(trades),
        "gate_counts": counts,
        "dynamic_sizing": sizing_summary,
    }

    out = Path("artifacts/official-candidate-oos-2024")
    out.mkdir(parents=True, exist_ok=True)
    sized.to_csv(out / "trades_2024.csv", index=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
