from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decision.trade_decision import TradeDecision
from market.market_context import MarketContext


@dataclass(slots=True)
class SignalScore:
    """
    Representa a qualidade esperada de um sinal.

    Este objeto será utilizado por:

    - Execution Plan
    - Risk Manager
    - Replay
    - Optimizer
    - Learning Database
    """

    value: float

    confidence: float

    quality: str

    recommendation: str

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_excellent(self) -> bool:
        return self.quality == "EXCELLENT"

    @property
    def is_good(self) -> bool:
        return self.quality == "GOOD"

    @property
    def is_average(self) -> bool:
        return self.quality == "AVERAGE"

    @property
    def is_poor(self) -> bool:
        return self.quality == "POOR"


class SignalScoreCalculator:
    """
    Calcula a qualidade geral do sinal.
    """

    def calculate(
        self,
        context: MarketContext,
        decision: TradeDecision,
    ) -> SignalScore:

        score = float(decision.institutional_score)

        confidence = float(decision.confidence)

        value = (
            score * 0.70 +
            confidence * 100 * 0.30
        )

        value = round(value, 2)

        if value >= 90:

            quality = "EXCELLENT"

            recommendation = "TRADE"

        elif value >= 80:

            quality = "GOOD"

            recommendation = "TRADE"

        elif value >= 65:

            quality = "AVERAGE"

            recommendation = "CAUTION"

        else:

            quality = "POOR"

            recommendation = "NO_TRADE"

        return SignalScore(

            value=value,

            confidence=confidence,

            quality=quality,

            recommendation=recommendation,

            metadata={

                "institutional_score": decision.institutional_score,

                "market_regime": context.regime,

                "trend": context.trend,

                "volatility": context.volatility,

                "direction": context.direction,
            },
        )