from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decision.trade_setup import TradeSetup


@dataclass(slots=True)
class ExposureResult:
    """
    Resultado da validação de exposição.
    """

    approved: bool

    reason: str

    current_positions: int

    max_positions: int

    exposure_ratio: float

    metadata: dict[str, Any] = field(default_factory=dict)


class ExposureManager:
    """
    Controla a exposição máxima permitida.

    Nesta primeira versão controla apenas o número
    máximo de posições simultâneas.

    Futuramente será expandido para:

    - exposição financeira
    - exposição por ativo
    - exposição por direção
    - correlação
    - alavancagem
    """

    def __init__(
        self,
        max_positions: int = 1,
    ):

        self.max_positions = max_positions

    def validate(
        self,
        setup: TradeSetup,
        current_positions: int = 0,
    ) -> ExposureResult:

        approved = (
            current_positions < self.max_positions
        )

        ratio = (
            current_positions / self.max_positions
            if self.max_positions > 0
            else 1.0
        )

        if approved:

            return ExposureResult(

                approved=True,

                reason="Exposure approved.",

                current_positions=current_positions,

                max_positions=self.max_positions,

                exposure_ratio=round(ratio, 2),

                metadata={

                    "direction": setup.direction,

                    "entry_strategy": setup.entry_strategy,

                    "signal_quality": setup.signal_quality,
                },
            )

        return ExposureResult(

            approved=False,

            reason="Maximum exposure reached.",

            current_positions=current_positions,

            max_positions=self.max_positions,

            exposure_ratio=round(ratio, 2),

            metadata={

                "direction": setup.direction,

                "entry_strategy": setup.entry_strategy,

                "signal_quality": setup.signal_quality,
            },
        )