from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CryptoLeverageStressConfig:
    oi_weight: float = 0.30
    funding_weight: float = 0.20
    premium_weight: float = 0.20
    positioning_weight: float = 0.15
    cvd_weight: float = 0.15
    clip_z: float = 3.0
    min_periods: int = 48
    rolling_window: int = 96


class CryptoLeverageStressScore:
    """Causal crypto-native leverage/crowding score.

    Positive values describe crowded/pressured LONG conditions.
    Negative values describe crowded/pressured SHORT conditions.

    The score deliberately uses rolling z-scores with only information
    available at or before each timestamp. It is not a trade signal by itself;
    it is intended for shadow-mode research and squeeze/deleveraging analysis.
    """

    def __init__(self, config: CryptoLeverageStressConfig | None = None):
        self.config = config or CryptoLeverageStressConfig()

    def _z(self, series: pd.Series) -> pd.Series:
        x = pd.to_numeric(series, errors="coerce")
        mean = x.rolling(self.config.rolling_window, min_periods=self.config.min_periods).mean()
        std = x.rolling(self.config.rolling_window, min_periods=self.config.min_periods).std(ddof=0)
        z = (x - mean) / std.replace(0, np.nan)
        return z.clip(-self.config.clip_z, self.config.clip_z) / self.config.clip_z

    @staticmethod
    def _ratio_pressure(frame: pd.DataFrame) -> pd.Series:
        cols = [
            "global_ls_ratio",
            "top_account_ls_ratio",
            "top_position_ls_ratio",
        ]
        available = [c for c in cols if c in frame.columns]
        if not available:
            return pd.Series(np.nan, index=frame.index)
        logs = pd.concat(
            [np.log(pd.to_numeric(frame[c], errors="coerce").clip(lower=1e-9)) for c in available],
            axis=1,
        )
        return logs.mean(axis=1)

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        c = frame.copy()
        required = ["open_interest_change_pct", "funding_rate", "premium_close"]
        missing = [name for name in required if name not in c.columns]
        if missing:
            raise ValueError(f"Missing leverage-stress inputs: {missing}")

        positioning_raw = self._ratio_pressure(c)
        oi_z = self._z(c["open_interest_change_pct"])
        funding_z = self._z(c["funding_rate"])
        premium_z = self._z(c["premium_close"])
        positioning_z = self._z(positioning_raw)

        if "cvd_delta" in c.columns:
            cvd_z = self._z(c["cvd_delta"])
        elif "cvd" in c.columns:
            cvd_z = self._z(c["cvd"].diff())
        else:
            cvd_z = pd.Series(np.nan, index=c.index)

        weighted = pd.concat(
            [
                oi_z * self.config.oi_weight,
                funding_z * self.config.funding_weight,
                premium_z * self.config.premium_weight,
                positioning_z * self.config.positioning_weight,
                cvd_z * self.config.cvd_weight,
            ],
            axis=1,
        )
        total_weight = pd.concat(
            [
                oi_z.notna() * self.config.oi_weight,
                funding_z.notna() * self.config.funding_weight,
                premium_z.notna() * self.config.premium_weight,
                positioning_z.notna() * self.config.positioning_weight,
                cvd_z.notna() * self.config.cvd_weight,
            ],
            axis=1,
        ).sum(axis=1)

        score = weighted.sum(axis=1, min_count=1) / total_weight.replace(0, np.nan)
        score = score.clip(-1.0, 1.0)

        c["leverage_oi_component"] = oi_z
        c["leverage_funding_component"] = funding_z
        c["leverage_premium_component"] = premium_z
        c["leverage_positioning_component"] = positioning_z
        c["leverage_cvd_component"] = cvd_z
        c["crypto_leverage_stress_score"] = score

        c["leverage_regime"] = "NEUTRAL"
        c.loc[score >= 0.50, "leverage_regime"] = "LONG_CROWDED"
        c.loc[score <= -0.50, "leverage_regime"] = "SHORT_CROWDED"

        price_delta = pd.to_numeric(c.get("close", np.nan), errors="coerce").pct_change()
        oi_delta = pd.to_numeric(c["open_interest_change_pct"], errors="coerce")

        c["deleveraging_regime"] = "NONE"
        c.loc[(price_delta > 0) & (oi_delta < 0), "deleveraging_regime"] = "SHORT_SQUEEZE_CANDIDATE"
        c.loc[(price_delta < 0) & (oi_delta < 0), "deleveraging_regime"] = "LONG_SQUEEZE_CANDIDATE"
        c.loc[(price_delta > 0) & (oi_delta > 0), "deleveraging_regime"] = "NEW_LONG_RISK"
        c.loc[(price_delta < 0) & (oi_delta > 0), "deleveraging_regime"] = "NEW_SHORT_RISK"
        return c
