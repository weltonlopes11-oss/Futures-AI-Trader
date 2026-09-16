from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiquidationPressureConfig:
    rolling_minutes: int = 15
    min_periods: int = 5


class LiquidationPressure:
    """Aggregate forward-collected forced-order events into causal features.

    Binance force-order SELL events are treated as long-liquidation notional;
    BUY events as short-liquidation notional. The class only consumes events
    observed at or before each output timestamp.
    """

    def __init__(self, config: LiquidationPressureConfig | None = None):
        self.config = config or LiquidationPressureConfig()

    def aggregate(self, events: pd.DataFrame, frequency: str = "1min") -> pd.DataFrame:
        required = {"timestamp", "side", "price", "quantity"}
        missing = required.difference(events.columns)
        if missing:
            raise ValueError(f"missing liquidation columns: {sorted(missing)}")
        if events.empty:
            return pd.DataFrame()

        e = events.copy()
        e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
        e["price"] = pd.to_numeric(e["price"], errors="coerce")
        e["quantity"] = pd.to_numeric(e["quantity"], errors="coerce")
        if e[["price", "quantity"]].isna().any().any():
            raise ValueError("liquidation events contain invalid numbers")
        if (e["price"] <= 0).any() or (e["quantity"] < 0).any():
            raise ValueError("invalid liquidation price/quantity")

        e["notional"] = e["price"] * e["quantity"]
        e["long_liq"] = np.where(e["side"].astype(str).str.upper() == "SELL", e["notional"], 0.0)
        e["short_liq"] = np.where(e["side"].astype(str).str.upper() == "BUY", e["notional"], 0.0)
        e = e.set_index("timestamp").sort_index()

        out = e[["long_liq", "short_liq"]].resample(frequency).sum()
        out["total_liq"] = out["long_liq"] + out["short_liq"]
        out["liquidation_imbalance"] = (
            (out["short_liq"] - out["long_liq"]) / out["total_liq"].replace(0, np.nan)
        ).fillna(0.0)

        window = self.config.rolling_minutes
        minp = self.config.min_periods
        out["long_liq_roll"] = out["long_liq"].rolling(window, min_periods=minp).sum()
        out["short_liq_roll"] = out["short_liq"].rolling(window, min_periods=minp).sum()
        out["total_liq_roll"] = out["total_liq"].rolling(window, min_periods=minp).sum()
        out["liq_imbalance_roll"] = (
            (out["short_liq_roll"] - out["long_liq_roll"])
            / out["total_liq_roll"].replace(0, np.nan)
        ).fillna(0.0)
        return out.reset_index()

    @staticmethod
    def classify(frame: pd.DataFrame) -> pd.DataFrame:
        c = frame.copy()
        if "liq_imbalance_roll" not in c.columns:
            raise ValueError("liq_imbalance_roll required")
        c["liquidation_regime"] = "BALANCED"
        c.loc[c["liq_imbalance_roll"] >= 0.50, "liquidation_regime"] = "SHORT_LIQUIDATION_PRESSURE"
        c.loc[c["liq_imbalance_roll"] <= -0.50, "liquidation_regime"] = "LONG_LIQUIDATION_PRESSURE"
        return c
