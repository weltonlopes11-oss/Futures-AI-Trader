from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, metrics
from run_regime_discovery_2025_validate_2026 import run_featured_regime_window

ROBUST_REGIMES = {
    "LONG|LOW|CHOP",
    "SHORT|NORMAL|MIXED",
}

INITIAL_EQUITY_BRL = 2000.0
MAX_LEVERAGE = 15.0
RISK_FRACTION = 0.10


def apply_continuous_equity_sizing(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = trades.copy().sort_values("entry_time").reset_index(drop=True)
    if out.empty:
        return out, {
            "initial_equity_brl": INITIAL_EQUITY_BRL,
            "ending_equity_brl": INITIAL_EQUITY_BRL,
            "total_pnl_brl": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    equity = INITIAL_EQUITY_BRL
    peak = equity
    max_dd_pct = 0.0
    max_dd_brl = 0.0

    rows = []
    for idx, row in out.iterrows():
        entry = float(row["entry_price"])
        stop = float(row["stop_price"])
        stop_distance_fraction = abs(entry - stop) / entry
        if not np.isfinite(stop_distance_fraction) or stop_distance_fraction <= 0:
            raise ValueError(f"Invalid stop distance at row {idx}")

        leverage = min(MAX_LEVERAGE, RISK_FRACTION / stop_distance_fraction)
        equity_before = equity
        notional = equity_before * leverage

        net_return_fraction = float(row["net_return_pct"]) / 100.0
        pnl = notional * net_return_fraction
        equity_after = equity_before + pnl

        peak = max(peak, equity_after)
        dd_brl = equity_after - peak
        dd_pct = (equity_after / peak - 1.0) * 100.0 if peak > 0 else float("nan")
        max_dd_brl = min(max_dd_brl, dd_brl)
        max_dd_pct = min(max_dd_pct, dd_pct)

        row_dict = row.to_dict()
        row_dict.update(
            {
                "equity_before_brl": equity_before,
                "risk_budget_brl": equity_before * RISK_FRACTION,
                "stop_distance_pct": stop_distance_fraction * 100.0,
                "sizing_leverage": leverage,
                "notional_brl": notional,
                "trade_pnl_brl": pnl,
                "equity_after_brl": equity_after,
                "drawdown_pct_after_trade": dd_pct,
            }
        )
        rows.append(row_dict)
        equity = equity_after

    sized = pd.DataFrame(rows)

    by_year = {}
    sized["exit_year"] = pd.to_datetime(sized["exit_time"]).dt.year
    for year, group in sized.groupby("exit_year", sort=True):
        start_equity = float(group.iloc[0]["equity_before_brl"])
        end_equity = float(group.iloc[-1]["equity_after_brl"])
        by_year[str(int(year))] = {
            "trades": int(len(group)),
            "start_equity_brl": start_equity,
            "end_equity_brl": end_equity,
            "pnl_brl": end_equity - start_equity,
            "return_on_year_start_pct": (end_equity / start_equity - 1.0) * 100.0,
            "simple_trade_return_pct": float(group["net_return_pct"].sum()),
            "wins": int((group["trade_pnl_brl"] > 0).sum()),
            "losses": int((group["trade_pnl_brl"] < 0).sum()),
        }

    by_month = {}
    sized["exit_month"] = pd.to_datetime(sized["exit_time"]).dt.strftime("%Y-%m")
    for month, group in sized.groupby("exit_month", sort=True):
        start_equity = float(group.iloc[0]["equity_before_brl"])
        end_equity = float(group.iloc[-1]["equity_after_brl"])
        by_month[month] = {
            "trades": int(len(group)),
            "start_equity_brl": start_equity,
            "end_equity_brl": end_equity,
            "pnl_brl": end_equity - start_equity,
            "return_pct": (end_equity / start_equity - 1.0) * 100.0,
        }

    summary = {
        "initial_equity_brl": INITIAL_EQUITY_BRL,
        "ending_equity_brl": float(equity),
        "total_pnl_brl": float(equity - INITIAL_EQUITY_BRL),
        "return_pct": float((equity / INITIAL_EQUITY_BRL - 1.0) * 100.0),
        "risk_fraction_of_current_equity": RISK_FRACTION,
        "max_leverage": MAX_LEVERAGE,
        "min_leverage_used": float(sized["sizing_leverage"].min()),
        "avg_leverage_used": float(sized["sizing_leverage"].mean()),
        "max_leverage_used": float(sized["sizing_leverage"].max()),
        "max_drawdown_brl": float(max_dd_brl),
        "max_drawdown_pct": float(max_dd_pct),
        "max_single_loss_brl": float(sized["trade_pnl_brl"].min()),
        "max_single_gain_brl": float(sized["trade_pnl_brl"].max()),
        "by_year": by_year,
        "by_month": by_month,
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

    trades, counts = run_featured_regime_window(
        h1,
        h4,
        eval_start,
        eval_end,
        cfg,
        allowed_regimes=ROBUST_REGIMES,
    )

    sized, sizing_summary = apply_continuous_equity_sizing(trades)

    result = {
        "experiment_id": "continuous-equity-risk-sizing-2024-2026",
        "status": "research_only_not_promoted",
        "official_strategy_modified": False,
        "period": "2024-01-01 through 2026-08-31 UTC",
        "rules_frozen_before_test": True,
        "rules": {
            "symbol": "ETHUSDT",
            "entry_timeframe": "1h",
            "direction_timeframe": "4h",
            "allowed_regimes": sorted(ROBUST_REGIMES),
            "stop": "2x ATR20",
            "target": "Keltner executable band",
            "fees_bps_per_side": 4.0,
            "slippage_bps_per_side": 1.0,
            "initial_equity_brl": INITIAL_EQUITY_BRL,
            "position_base": "100% of current available equity before each trade",
            "risk_budget": "10% of current available equity at the protective stop",
            "max_leverage": MAX_LEVERAGE,
            "sizing_formula": "leverage = min(15x, 0.10 / stop_distance_fraction)",
            "notional_formula": "current_equity * leverage",
            "equity_update": "current_equity + trade_pnl; full equity becomes next trade base",
        },
        "trade_metrics_underlying": metrics(trades),
        "gate_counts": counts,
        "continuous_equity_sizing": sizing_summary,
    }

    out = Path("artifacts/continuous-equity-risk-sizing-2024-2026")
    out.mkdir(parents=True, exist_ok=True)
    sized.to_csv(out / "trades_continuous_2024_to_aug2026.csv", index=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
