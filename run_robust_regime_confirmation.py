from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.binance_data_vision import BinanceDataVisionLoader
from backtest.ichimoku_keltner_1h import IchimokuKeltner1HConfig, metrics
from run_regime_4h_ema50_backtest import leveraged_path
from run_regime_discovery_2025_validate_2026 import run_featured_regime_window

ROBUST_REGIMES = {
    "LONG|LOW|CHOP",
    "SHORT|NORMAL|MIXED",
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

    out = Path("artifacts/robust-regimes")
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "research_only_not_promoted",
        "official_strategy_modified": False,
        "robust_regimes": sorted(ROBUST_REGIMES),
        "periods": {},
    }

    for name, (pstart, pend) in periods.items():
        trades, counts = run_featured_regime_window(
            h1, h4, pstart, pend, cfg, allowed_regimes=ROBUST_REGIMES
        )
        result["periods"][name] = {
            "metrics": metrics(trades),
            "leveraged": leveraged_path(trades),
            "counts": counts,
        }
        trades.to_csv(out / f"{name.lower()}_trades.csv", index=False)

    (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
