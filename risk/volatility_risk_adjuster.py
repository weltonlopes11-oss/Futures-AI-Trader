from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from market.market_context import MarketContext


@dataclass(slots=True)
class VolatilityRiskResult:
    """
    Resultado do ajuste de risco baseado
    na volatilidade do mercado.
    """

    approved: bool

    allowed_risk_percent: float

    volatility: str

    adjustment_factor: float

    reason: str

    metadata: dict[str, Any] = field(default_factory=dict)


class VolatilityRiskAdjuster:
    """
    Ajusta o risco permitido conforme
    a volatilidade do mercado.

    Não calcula o risco da operação.

    Apenas informa qual percentual máximo
    de risco deve ser utilizado.
    """

    def __init__(
        self,
        low_risk: float = 1.20,
        normal_risk: float = 1.00,
        high_risk: float = 0.50,
    ):

        self.low_risk = low_risk
        self.normal_risk = normal_risk
        self.high_risk = high_risk

    def validate(
        self,
        context: MarketContext,
    ) -> VolatilityRiskResult:

        volatility = context.volatility

        if volatility == "LOW":

            return VolatilityRiskResult(

                approved=True,

                allowed_risk_percent=self.low_risk,

                volatility=volatility,

                adjustment_factor=1.20,

                reason="Low volatility allows increased risk.",

                metadata={

                    "market_regime": context.regime,

                    "confidence": context.confidence,
                },
            )

        if volatility == "NORMAL":

            return VolatilityRiskResult(

                approved=True,

                allowed_risk_percent=self.normal_risk,

                volatility=volatility,

                adjustment_factor=1.00,

                reason="Normal volatility.",

                metadata={

                    "market_regime": context.regime,

                    "confidence": context.confidence,
                },
            )

        return VolatilityRiskResult(

            approved=True,

            allowed_risk_percent=self.high_risk,

            volatility="HIGH",

            adjustment_factor=0.50,

            reason="High volatility. Risk reduced.",

            metadata={

                "market_regime": context.regime,

                "confidence": context.confidence,
            },
        )