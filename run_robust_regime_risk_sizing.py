from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig
from run_regime_discovery_2025_validate_2026 import run_featured_regime_window

INITIAL_CAPITAL_BRL = 2000.0
MAX_LEVERAGE = 10.0
RISK_LEVELS = [0.01, 0.02, 0.03, 0.05]
ROBUST_REGIMES = {
    "LONG|LOW|CHOP",
    "SHORT|NORMAL|MIXED",
}


def simulate_fixed_fractional_risk(
    trades: pd.DataFrame,
    risk_fraction: float,
    initial_capital: float = INITIAL_CAPITAL_BRL,
    max_leverage: float = MAX_LEVERAGE,
) -> dict:
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    rows = []

    for _, trade in trades.sort_values("entry_time").iterrows():
        entry = float(trade["entry_price"])
        atr = float(trade["entry_atr"])
        net_ret_frac = float(trade["net_return_pct"]) / 100.0

        stop_distance_frac = (2.0 * atr) / entry
        raw_multiple = risk_fraction / stop_distance_frac if stop_distance_frac > 0 else 0.0
        notional_multiple = min(raw_multiple, max_leverage)

        capital_before = equity
        pnl = equity * notional_multiple * net_ret_frac
        equity += pnl

        peak = max(peak, equity)
        dd = (equity / peak - 1.0) * 100.0
        max_dd = min(max_dd, dd)

        planned_stop_risk_pct = notional_multiple * stop_distance_frac * 100.0

        rows.append(
            {
                "entry_time": trade["entry_time"],
                "exit_time": trade["exit_time"],
                "side": trade["side"],
                "regime_key": trade["regime_key"],
                "entry_price": entry,
                "entry_atr": atr,
                "stop_distance_pct": stop_distance_frac * 100.0,
                "notional_multiple": notional_multiple,
                "planned_stop_risk_pct": planned_stop_risk_pct,
                "trade_net_return_pct_on_notional": float(trade["net_return_pct"]),
                "capital_before_brl": capital_before,
                "pnl_brl": pnl,
                "capital_after_brl": equity,
                "drawdown_pct": dd,
                "exit_reason": trade["exit_reason"],
            }
        )

    return {
        "initial_brl": initial_capital,
        "final_brl": equity,
        "return_pct": (equity / initial_capital - 1.0) * 100.0,
        "max_drawdown_pct": max_dd,
        "trades": len(rows),
        "avg_notional_multiple": (
            sum(r["notional_multiple"] for r in rows) / len(rows) if rows else 0.0
        ),
        "max_notional_multiple": max((r["notional_multiple"] for r in rows), default=0.0),
        "min_notional_multiple": min((r["notional_multiple"] for r in rows), default=0.0),
        "rows": rows,
    }


def main() -> None:
    cfg = IchimokuKeltner1HConfig(fee_bps_per_side=4.0, slippage_bps_per_side=1.0)

    periods = {
        "2025": (
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        "2026_JAN_AUG": (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 1, tzinfo=timezone.utc),
        ),
    }

    loader = BinanceDataVisionLoader()
    start = periods["2025"][0]
    end = periods["2026_JAN_AUG"][1]
    warmup = start - timedelta(days=120)

    h1 = loader.fetch_window("ETHUSDT", "1h", warmup, end)
    h4 = loader.fetch_window("ETHUSDT", "4h", warmup - timedelta(days=15), end)

    out = Path("artifacts/risk-sizing")
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "research_only_not_promoted",
        "official_strategy_modified": False,
        "robust_regimes": sorted(ROBUST_REGIMES),
        "risk_model": {
            "type": "fixed_fractional_from_2ATR_stop",
            "formula": "notional_multiple = min(max_leverage, risk_fraction / (2*ATR/entry_price))",
            "max_leverage": MAX_LEVERAGE,
            "risk_levels": RISK_LEVELS,
            "capital_initial_brl": INITIAL_CAPITAL_BRL,
            "fees_and_slippage": "already embedded in trade net returns",
            "note": "gap exits can exceed planned stop risk; realized PnL uses actual trade result",
        },
        "periods": {},
    }

    for period_name, (pstart, pend) in periods.items():
        trades, _ = run_featured_regime_window(
            h1, h4, pstart, pend, cfg, allowed_regimes=ROBUST_REGIMES
        )
        period_result = {}
        for risk in RISK_LEVELS:
            sim = simulate_fixed_fractional_risk(trades, risk)
            key = f"{int(risk*100)}pct_risk"
            pd.DataFrame(sim["rows"]).to_csv(
                out / f"{period_name.lower()}_{key}_trades.csv", index=False
            )
            period_result[key] = {k: v for k, v in sim.items() if k != "rows"}

        result["periods"][period_name] = period_result

    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
