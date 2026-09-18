from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, metrics
from run_ichimoku_keltner_1h_stop_backtest import run_stop_window
from run_regime_4h_ema50_backtest import leveraged_path, run_regime_window
import run_regime_4h_atr_price_max_backtest as atr_mod

THRESHOLDS = [0.008, 0.010, 0.012, 0.015]


def main() -> None:
    symbol = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
    eval_start = datetime.fromisoformat(
        os.getenv("BACKTEST_EVAL_START_UTC", "2026-01-01T00:00:00+00:00").replace("Z", "+00:00")
    )
    eval_end = datetime.fromisoformat(
        os.getenv("BACKTEST_EVAL_END_UTC", "2026-09-01T00:00:00+00:00").replace("Z", "+00:00")
    )
    if eval_start.tzinfo is None:
        eval_start = eval_start.replace(tzinfo=timezone.utc)
    if eval_end.tzinfo is None:
        eval_end = eval_end.replace(tzinfo=timezone.utc)

    cfg = IchimokuKeltner1HConfig(
        fee_bps_per_side=float(os.getenv("BACKTEST_FEE_BPS_PER_SIDE", "4")),
        slippage_bps_per_side=float(os.getenv("BACKTEST_SLIPPAGE_BPS_PER_SIDE", "1")),
    )

    loader = BinanceDataVisionLoader()
    warmup_start = eval_start - timedelta(days=30)
    h1 = loader.fetch_window(symbol, "1h", warmup_start, eval_end)
    h4 = loader.fetch_window(symbol, "4h", warmup_start - timedelta(days=15), eval_end)

    official = run_stop_window(h1, eval_start, eval_end, cfg)
    regime_only, regime_counts = run_regime_window(h1, h4, eval_start, eval_end, cfg)

    variants = {}
    out = Path("artifacts/regime-4h-atr-price-robustness")
    out.mkdir(parents=True, exist_ok=True)

    for threshold in THRESHOLDS:
        atr_mod.ATR_PRICE_MAX = threshold
        trades, counts = atr_mod.run_regime_atr_price_window(h1, h4, eval_start, eval_end, cfg)
        key = f"{threshold * 100:.1f}%"
        variants[key] = {
            "threshold": threshold,
            "metrics": metrics(trades),
            "leveraged_path": leveraged_path(trades),
            "gate_counts": counts,
            "exit_reason_counts": trades["exit_reason"].value_counts().to_dict() if not trades.empty else {},
        }
        trades.to_csv(out / f"trades_atr_price_{int(round(threshold * 10000)):03d}bp.csv", index=False)

    result = {
        "experiment_id": "4h-ema50-atr-price-robustness-jan-aug-2026",
        "status": "separate_experiment_not_promoted",
        "official_strategy_modified": False,
        "period": "2026-01-01 through 2026-08-31 UTC",
        "purpose": "sensitivity test only; not threshold optimization",
        "thresholds_tested": THRESHOLDS,
        "official": metrics(official),
        "regime_4h_only": metrics(regime_only),
        "regime_4h_only_leveraged_path": leveraged_path(regime_only),
        "regime_4h_only_counts": regime_counts,
        "variants": variants,
    }

    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
