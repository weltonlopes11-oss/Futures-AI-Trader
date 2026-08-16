from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from features.atr_calculator import ATRCalculator


@dataclass(slots=True)
class VolatilityResult:
    """
    Resultado da classificação da volatilidade.
    """

    volatility: str
    atr_value: float
    atr_percent: float
    confidence: float
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VolatilityDetector:
    """
    Classifica a volatilidade do mercado utilizando ATR.

    Classificações:

        LOW
        NORMAL
        HIGH

    O detector também retorna:

    - valor do ATR
    - ATR percentual
    - score
    - confidence
    """

    def __init__(
        self,
        atr_period: int = 14,
        low_threshold: float = 0.8,
        high_threshold: float = 1.2,
    ):

        self.atr = ATRCalculator(period=atr_period)

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def detect(self, data: pd.DataFrame) -> VolatilityResult:

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "VolatilityDetector espera um DataFrame."
            )

        if data.empty:
            raise ValueError("DataFrame vazio.")

        df = self.atr.calculate(data)

        latest = df.iloc[-1]

        atr = float(latest["atr"])

        close = float(latest["close"])

        if close <= 0:
            raise ValueError("Preço inválido.")

        atr_percent = (atr / close) * 100

        historical = (
            df["atr"]
            .dropna()
            .tail(100)
        )

        if historical.empty:

            average = atr

        else:

            average = historical.mean()

        ratio = atr / average if average > 0 else 1.0

        metadata = {
            "atr": atr,
            "atr_percent": atr_percent,
            "atr_average": average,
            "ratio": ratio,
        }

        # -------------------------
        # HIGH
        # -------------------------

        if ratio >= self.high_threshold:

            return VolatilityResult(
                volatility="HIGH",
                atr_value=atr,
                atr_percent=atr_percent,
                confidence=min(ratio / 2.0, 1.0),
                score=95,
                metadata=metadata,
            )

        # -------------------------
        # LOW
        # -------------------------

        if ratio <= self.low_threshold:

            confidence = min(
                (1.0 - ratio),
                1.0,
            )

            return VolatilityResult(
                volatility="LOW",
                atr_value=atr,
                atr_percent=atr_percent,
                confidence=confidence,
                score=70,
                metadata=metadata,
            )

        # -------------------------
        # NORMAL
        # -------------------------

        confidence = 1.0 - abs(1.0 - ratio)

        return VolatilityResult(
            volatility="NORMAL",
            atr_value=atr,
            atr_percent=atr_percent,
            confidence=max(confidence, 0.5),
            score=85,
            metadata=metadata,
        )