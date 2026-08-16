from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class TrendResult:
    trend: str
    direction: str
    confidence: float
    score: float
    metadata: dict


class TrendDetector:
    """
    Detecta a tendência predominante utilizando
    os três timeframes principais.

    Espera encontrar no último candle:

        trend_1d
        trend_4h
        trend_1h

    Valores possíveis:

        BULL
        BEAR
        RANGE
    """

    VALID_VALUES = {
        "BULL",
        "BEAR",
        "RANGE",
    }

    def detect(self, data: pd.DataFrame) -> TrendResult:

        if not isinstance(data, pd.DataFrame):
            raise TypeError("TrendDetector espera um DataFrame.")

        if data.empty:
            raise ValueError("DataFrame vazio.")

        last = data.iloc[-1]

        required = [
            "trend_1d",
            "trend_4h",
            "trend_1h",
        ]

        for column in required:
            if column not in last.index:
                raise ValueError(f"Coluna ausente: {column}")

        trend_1d = str(last["trend_1d"]).upper()
        trend_4h = str(last["trend_4h"]).upper()
        trend_1h = str(last["trend_1h"]).upper()

        trends = [
            trend_1d,
            trend_4h,
            trend_1h,
        ]

        for value in trends:
            if value not in self.VALID_VALUES:
                raise ValueError(
                    f"Tendência inválida: {value}"
                )

        bull = trends.count("BULL")
        bear = trends.count("BEAR")
        ranging = trends.count("RANGE")

        metadata = {
            "trend_1d": trend_1d,
            "trend_4h": trend_4h,
            "trend_1h": trend_1h,
            "bull_count": bull,
            "bear_count": bear,
            "range_count": ranging,
        }

        # Forte alta

        if bull == 3:
            return TrendResult(
                trend="STRONG_BULL",
                direction="LONG",
                confidence=1.00,
                score=100,
                metadata=metadata,
            )

        # Alta

        if bull == 2:
            return TrendResult(
                trend="WEAK_BULL",
                direction="LONG",
                confidence=0.82,
                score=85,
                metadata=metadata,
            )

        # Forte baixa

        if bear == 3:
            return TrendResult(
                trend="STRONG_BEAR",
                direction="SHORT",
                confidence=1.00,
                score=100,
                metadata=metadata,
            )

        # Baixa

        if bear == 2:
            return TrendResult(
                trend="WEAK_BEAR",
                direction="SHORT",
                confidence=0.82,
                score=85,
                metadata=metadata,
            )

        # Mercado lateral

        return TrendResult(
            trend="RANGE",
            direction="NEUTRAL",
            confidence=0.50,
            score=50,
            metadata=metadata,
        )