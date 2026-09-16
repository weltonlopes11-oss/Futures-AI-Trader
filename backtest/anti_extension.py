from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AntiExtensionConfig:
    max_vwap_distance_atr: float = 2.0
    max_keltner_position: float = 0.75


class AntiExtensionFilter:
    """Directional anti-chasing filter using already-causal VWAP/Keltner features.

    LONG extension is positive when price is above VWAP / toward the upper
    Keltner side. SHORT extension is mirrored so positive always means
    stretched in the intended trade direction.
    """

    def __init__(self, config: AntiExtensionConfig | None = None):
        self.config = config or AntiExtensionConfig()
        if self.config.max_vwap_distance_atr <= 0:
            raise ValueError("max_vwap_distance_atr must be positive")
        if self.config.max_keltner_position <= 0:
            raise ValueError("max_keltner_position must be positive")

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"decision", "vwap_distance_atr", "keltner_position"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"anti-extension missing columns: {sorted(missing)}")

        out = frame.copy()
        decision = out["decision"].astype(str).str.upper()
        side_sign = decision.map({"LONG": 1.0, "SHORT": -1.0}).fillna(0.0)

        out["directional_vwap_extension_atr"] = pd.to_numeric(
            out["vwap_distance_atr"], errors="coerce"
        ) * side_sign
        out["directional_keltner_extension"] = pd.to_numeric(
            out["keltner_position"], errors="coerce"
        ) * side_sign

        out["vwap_overextended"] = (
            out["directional_vwap_extension_atr"] > self.config.max_vwap_distance_atr
        )
        out["keltner_overextended"] = (
            out["directional_keltner_extension"] > self.config.max_keltner_position
        )
        return out

    def apply(self, frame: pd.DataFrame, mode: str) -> pd.DataFrame:
        out = self.enrich(frame)
        active = out["decision"].astype(str).str.upper().isin(["LONG", "SHORT"])

        if mode == "control":
            reject = pd.Series(False, index=out.index)
        elif mode == "vwap_only":
            reject = active & out["vwap_overextended"]
        elif mode == "keltner_only":
            reject = active & out["keltner_overextended"]
        elif mode == "combined_either":
            reject = active & (out["vwap_overextended"] | out["keltner_overextended"])
        elif mode == "combined_both":
            reject = active & (out["vwap_overextended"] & out["keltner_overextended"])
        else:
            raise ValueError(f"unknown anti-extension mode: {mode}")

        out["anti_extension_rejected"] = reject
        out.loc[reject, "decision"] = "NO_TRADE"
        return out
