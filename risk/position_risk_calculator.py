from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PositionRiskResult:
    """
    Resultado do cálculo de risco da posição.
    """

    risk_amount: float

    risk_percent: float

    stop_distance: float

    leverage: float

    position_size: float

    approved: bool

    metadata: dict[str, Any] = field(default_factory=dict)


class PositionRiskCalculator:
    """
    Calcula o risco financeiro de uma operação.

    Este módulo não define o tamanho da posição.

    Apenas informa quanto risco a operação
    representa.
    """

    def __init__(
        self,
        max_risk_percent: float = 1.0,
    ):

        self.max_risk_percent = max_risk_percent

    def calculate(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        position_size: float,
        leverage: float = 1.0,
    ) -> PositionRiskResult:

        if equity <= 0:
            raise ValueError(
                "Equity deve ser maior que zero."
            )

        if entry_price <= 0:
            raise ValueError(
                "Preço de entrada inválido."
            )

        if stop_price <= 0:
            raise ValueError(
                "Stop inválido."
            )

        if position_size <= 0:
            raise ValueError(
                "Position Size inválido."
            )

        if leverage <= 0:
            raise ValueError(
                "Alavancagem inválida."
            )

        stop_distance = abs(
            entry_price - stop_price
        )

        risk_amount = (
            stop_distance
            * position_size
        )

        risk_percent = (
            risk_amount
            / equity
        ) * 100

        approved = (
            risk_percent <= self.max_risk_percent
        )

        return PositionRiskResult(

            risk_amount=round(
                risk_amount,
                2,
            ),

            risk_percent=round(
                risk_percent,
                4,
            ),

            stop_distance=round(
                stop_distance,
                2,
            ),

            leverage=leverage,

            position_size=position_size,

            approved=approved,

            metadata={

                "entry_price": entry_price,

                "stop_price": stop_price,

                "max_risk_percent": self.max_risk_percent,

                "remaining_risk": max(
                    self.max_risk_percent
                    - risk_percent,
                    0.0,
                ),
            },
        )