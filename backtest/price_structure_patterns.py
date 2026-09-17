from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceStructureConfig:
    pivot_span: int = 3
    double_tolerance_atr: float = 0.25
    double_min_separation_bars: int = 4


class PriceStructurePatterns:
    """Causal price-structure features confirmed only with completed bars.

    A pivot at candidate bar i is confirmed only at i + pivot_span, after the
    required bars to the right have completed. Therefore the pivot price and
    its HH/HL/LH/LL label become available only on the confirmation bar.

    BOS/CHOCH are evaluated against the latest confirmed swing levels.
    Double tops/bottoms compare newly confirmed pivots with the previous pivot
    of the same type, using an ATR-normalized tolerance.
    """

    def __init__(self, config: PriceStructureConfig | None = None):
        self.config = config or PriceStructureConfig()
        if self.config.pivot_span < 1:
            raise ValueError("pivot_span must be >= 1")
        if self.config.double_tolerance_atr <= 0:
            raise ValueError("double_tolerance_atr must be positive")
        if self.config.double_min_separation_bars < 1:
            raise ValueError("double_min_separation_bars must be >= 1")

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "high", "low", "close", "atr"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"price structure missing columns: {sorted(missing)}")

        c = frame.copy().sort_values("timestamp").reset_index(drop=True)
        high = pd.to_numeric(c["high"], errors="coerce")
        low = pd.to_numeric(c["low"], errors="coerce")
        close = pd.to_numeric(c["close"], errors="coerce")
        atr = pd.to_numeric(c["atr"], errors="coerce")
        n = len(c)
        s = self.config.pivot_span

        c["pivot_high_confirmed"] = False
        c["pivot_low_confirmed"] = False
        c["confirmed_pivot_high_price"] = np.nan
        c["confirmed_pivot_low_price"] = np.nan
        c["confirmed_pivot_high_index"] = np.nan
        c["confirmed_pivot_low_index"] = np.nan
        c["swing_high_class"] = "NONE"
        c["swing_low_class"] = "NONE"
        c["double_top"] = False
        c["double_bottom"] = False

        prev_high_price = None
        prev_high_idx = None
        prev_low_price = None
        prev_low_idx = None

        for confirm_idx in range(2 * s, n):
            candidate_idx = confirm_idx - s
            left = candidate_idx - s
            right = candidate_idx + s
            window_high = high.iloc[left : right + 1]
            window_low = low.iloc[left : right + 1]
            candidate_high = high.iloc[candidate_idx]
            candidate_low = low.iloc[candidate_idx]

            is_high = pd.notna(candidate_high) and candidate_high == window_high.max() and int((window_high == candidate_high).sum()) == 1
            is_low = pd.notna(candidate_low) and candidate_low == window_low.min() and int((window_low == candidate_low).sum()) == 1

            if is_high:
                c.at[confirm_idx, "pivot_high_confirmed"] = True
                c.at[confirm_idx, "confirmed_pivot_high_price"] = float(candidate_high)
                c.at[confirm_idx, "confirmed_pivot_high_index"] = int(candidate_idx)
                if prev_high_price is not None:
                    c.at[confirm_idx, "swing_high_class"] = "HH" if candidate_high > prev_high_price else "LH"
                    sep = candidate_idx - int(prev_high_idx)
                    tol = self.config.double_tolerance_atr * float(atr.iloc[confirm_idx]) if pd.notna(atr.iloc[confirm_idx]) else np.nan
                    if sep >= self.config.double_min_separation_bars and pd.notna(tol) and abs(float(candidate_high) - float(prev_high_price)) <= tol:
                        c.at[confirm_idx, "double_top"] = True
                prev_high_price = float(candidate_high)
                prev_high_idx = int(candidate_idx)

            if is_low:
                c.at[confirm_idx, "pivot_low_confirmed"] = True
                c.at[confirm_idx, "confirmed_pivot_low_price"] = float(candidate_low)
                c.at[confirm_idx, "confirmed_pivot_low_index"] = int(candidate_idx)
                if prev_low_price is not None:
                    c.at[confirm_idx, "swing_low_class"] = "HL" if candidate_low > prev_low_price else "LL"
                    sep = candidate_idx - int(prev_low_idx)
                    tol = self.config.double_tolerance_atr * float(atr.iloc[confirm_idx]) if pd.notna(atr.iloc[confirm_idx]) else np.nan
                    if sep >= self.config.double_min_separation_bars and pd.notna(tol) and abs(float(candidate_low) - float(prev_low_price)) <= tol:
                        c.at[confirm_idx, "double_bottom"] = True
                prev_low_price = float(candidate_low)
                prev_low_idx = int(candidate_idx)

        # Forward-fill only confirmed information. A pivot cannot exist in state
        # before its confirmation bar.
        c["last_swing_high"] = c["confirmed_pivot_high_price"].ffill()
        c["last_swing_low"] = c["confirmed_pivot_low_price"].ffill()
        c["last_swing_high_index"] = c["confirmed_pivot_high_index"].ffill()
        c["last_swing_low_index"] = c["confirmed_pivot_low_index"].ffill()

        # Use prior-bar confirmed levels for break tests so a same-bar pivot
        # confirmation cannot redefine the level before the close is tested.
        ref_high = c["last_swing_high"].shift(1)
        ref_low = c["last_swing_low"].shift(1)
        c["bos_up"] = (close > ref_high) & (close.shift(1) <= ref_high.shift(1))
        c["bos_down"] = (close < ref_low) & (close.shift(1) >= ref_low.shift(1))

        # Structural regime is derived from the most recently confirmed swing
        # labels. CHOCH is the first break against that regime.
        high_state = c["swing_high_class"].replace("NONE", np.nan).ffill()
        low_state = c["swing_low_class"].replace("NONE", np.nan).ffill()
        bullish_structure = (high_state == "HH") & (low_state == "HL")
        bearish_structure = (high_state == "LH") & (low_state == "LL")
        c["structure_regime"] = np.select(
            [bullish_structure, bearish_structure],
            ["BULLISH", "BEARISH"],
            default="NEUTRAL",
        )
        prior_regime = c["structure_regime"].shift(1)
        c["choch_up"] = c["bos_up"] & (prior_regime == "BEARISH")
        c["choch_down"] = c["bos_down"] & (prior_regime == "BULLISH")

        c["distance_to_swing_high_atr"] = (c["last_swing_high"] - close) / atr.replace(0.0, np.nan)
        c["distance_to_swing_low_atr"] = (close - c["last_swing_low"]) / atr.replace(0.0, np.nan)
        c["bars_since_swing_high"] = np.where(
            c["last_swing_high_index"].notna(), c.index.to_numpy() - c["last_swing_high_index"], np.nan
        )
        c["bars_since_swing_low"] = np.where(
            c["last_swing_low_index"].notna(), c.index.to_numpy() - c["last_swing_low_index"], np.nan
        )

        return c
