from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decision.trade_setup import TradeSetup


@dataclass(slots=True)
class DailyLossResult:
    """
    Resultado da validação de perda diária.
    """

    approved: bool

    reason: str

    daily_loss: float

    max_daily_loss: float

    loss_ratio: float

    metadata: dict[str, Any] = field(default_factory=dict)


class DailyLossLimiter:
    """
    Impede novas operações quando o limite diário
    de perda for atingido.

    Nesta primeira versão recebe a perda diária
    como parâmetro.

    Futuramente esse valor será obtido através do
    PortfolioState.
    """

    def __init__(
        self,
        max_daily_loss: float = 500.0,
    ):

        self.max_daily_loss = max_daily_loss

    def validate(
        self,
        setup: TradeSetup,
        daily_loss: float = 0.0,
    ) -> DailyLossResult:

        daily_loss = max(0.0, daily_loss)

        approved = daily_loss < self.max_daily_loss

        if self.max_daily_loss > 0:

            ratio = daily_loss / self.max_daily_loss

        else:

            ratio = 1.0

        metadata = {

            "direction": setup.direction,

            "entry_strategy": setup.entry_strategy,

            "signal_quality": setup.signal_quality,

            "remaining_loss": max(
                self.max_daily_loss - daily_loss,
                0.0,
            ),
        }

        if approved:

            return DailyLossResult(

                approved=True,

                reason="Daily loss within limit.",

                daily_loss=round(daily_loss, 2),

                max_daily_loss=self.max_daily_loss,

                loss_ratio=round(ratio, 4),

                metadata=metadata,
            )

        return DailyLossResult(

            approved=False,

            reason="Maximum daily loss exceeded.",

            daily_loss=round(daily_loss, 2),

            max_daily_loss=self.max_daily_loss,

            loss_ratio=round(ratio, 4),

            metadata=metadata,
        )