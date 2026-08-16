from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MarginResult:
    """
    Resultado da validação de margem.
    """

    approved: bool

    reason: str

    available_margin: float

    required_margin: float

    remaining_margin: float

    margin_ratio: float

    metadata: dict[str, Any] = field(default_factory=dict)


class MarginChecker:
    """
    Valida se existe margem suficiente para abrir
    uma nova posição.

    Não conhece Binance.

    Recebe apenas os dados necessários para a
    validação.
    """

    def __init__(
        self,
        minimum_margin_buffer: float = 50.0,
    ):

        self.minimum_margin_buffer = minimum_margin_buffer

    def validate(
        self,
        available_margin: float,
        required_margin: float,
        leverage: float,
    ) -> MarginResult:

        if available_margin < 0:
            raise ValueError(
                "Margem disponível inválida."
            )

        if required_margin <= 0:
            raise ValueError(
                "Margem requerida inválida."
            )

        if leverage <= 0:
            raise ValueError(
                "Alavancagem inválida."
            )

        remaining_margin = (
            available_margin - required_margin
        )

        approved = (
            remaining_margin
            >= self.minimum_margin_buffer
        )

        margin_ratio = (
            required_margin / available_margin
            if available_margin > 0
            else 1.0
        )

        metadata = {

            "leverage": leverage,

            "minimum_margin_buffer":
                self.minimum_margin_buffer,

            "margin_utilization_percent":
                round(
                    margin_ratio * 100,
                    2,
                ),
        }

        if approved:

            return MarginResult(

                approved=True,

                reason="Margin approved.",

                available_margin=round(
                    available_margin,
                    2,
                ),

                required_margin=round(
                    required_margin,
                    2,
                ),

                remaining_margin=round(
                    remaining_margin,
                    2,
                ),

                margin_ratio=round(
                    margin_ratio,
                    4,
                ),

                metadata=metadata,
            )

        return MarginResult(

            approved=False,

            reason="Insufficient available margin.",

            available_margin=round(
                available_margin,
                2,
            ),

            required_margin=round(
                required_margin,
                2,
            ),

            remaining_margin=round(
                remaining_margin,
                2,
            ),

            margin_ratio=round(
                margin_ratio,
                4,
            ),

            metadata=metadata,
        )