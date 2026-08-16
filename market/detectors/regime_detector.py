from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from market.detectors.trend_detector import TrendDetector
from market.detectors.volatility_detector import VolatilityDetector


@dataclass(slots=True)
class RegimeResult:
    """
    Resultado consolidado do regime de mercado.
    """

    regime: str

    direction: str

    confidence: float

    score: float

    metadata: dict[str, Any] = field(default_factory=dict)


class RegimeDetector:
    """
    Consolida os detectores de mercado.

    Nesta primeira versão utiliza:

    - TrendDetector
    - VolatilityDetector

    Futuramente serão adicionados:

    - OI Detector
    - Funding Detector
    - CVD Detector
    - Liquidity Detector
    """

    def __init__(
        self,
        trend_detector: TrendDetector | None = None,
        volatility_detector: VolatilityDetector | None = None,
    ):

        self.trend_detector = (
            trend_detector or TrendDetector()
        )

        self.volatility_detector = (
            volatility_detector or VolatilityDetector()
        )

    def detect(
        self,
        data: pd.DataFrame,
    ) -> RegimeResult:

        trend = self.trend_detector.detect(data)

        volatility = self.volatility_detector.detect(data)

        metadata = {
            "trend": trend,
            "volatility": volatility,
        }

        # ---------------------------------------------------
        # STRONG BULL
        # ---------------------------------------------------

        if (
            trend.trend == "STRONG_BULL"
            and volatility.volatility != "HIGH"
        ):

            return RegimeResult(
                regime="STRONG_BULL",
                direction="LONG",
                confidence=0.98,
                score=100,
                metadata=metadata,
            )

        # ---------------------------------------------------
        # WEAK BULL
        # ---------------------------------------------------

        if trend.trend == "WEAK_BULL":

            return RegimeResult(
                regime="WEAK_BULL",
                direction="LONG",
                confidence=0.86,
                score=88,
                metadata=metadata,
            )

        # ---------------------------------------------------
        # STRONG BEAR
        # ---------------------------------------------------

        if (
            trend.trend == "STRONG_BEAR"
            and volatility.volatility != "HIGH"
        ):

            return RegimeResult(
                regime="STRONG_BEAR",
                direction="SHORT",
                confidence=0.98,
                score=100,
                metadata=metadata,
            )

        # ---------------------------------------------------
        # WEAK BEAR
        # ---------------------------------------------------

        if trend.trend == "WEAK_BEAR":

            return RegimeResult(
                regime="WEAK_BEAR",
                direction="SHORT",
                confidence=0.86,
                score=88,
                metadata=metadata,
            )

        # ---------------------------------------------------
        # ALTA VOLATILIDADE
        # ---------------------------------------------------

        if volatility.volatility == "HIGH":

            return RegimeResult(
                regime="HIGH_VOLATILITY",
                direction="NEUTRAL",
                confidence=0.92,
                score=75,
                metadata=metadata,
            )

        # ---------------------------------------------------
        # MERCADO LATERAL
        # ---------------------------------------------------

        return RegimeResult(
            regime="RANGE",
            direction="NEUTRAL",
            confidence=0.70,
            score=60,
            metadata=metadata,
        )