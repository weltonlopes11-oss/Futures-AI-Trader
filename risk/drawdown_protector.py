from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decision.trade_setup import TradeSetup


@dataclass(slots=True)
class DrawdownResult:
    """
    Resultado da validação de drawdown.
    """

    approved: bool

    reason: str

    peak_equity: float

    current_equity: float

    drawdown_percent: float

    max_drawdown_percent: float

    metadata: dict[str, Any] = field(default_factory=dict)


class DrawdownProtector:
    """
    Protege o robô contra drawdowns excessivos.

    O drawdown é calculado sempre em relação ao
    maior patrimônio (Peak Equity).

    Futuramente os valores serão obtidos através
    do PortfolioState.
    """

    def __init__(
        self,
        max_drawdown_percent: float = 8.0,
    ):

        self.max_drawdown_percent = max_drawdown_percent

    def validate(
        self,
        setup: TradeSetup,
        peak_equity: float,
        current_equity: float,
    ) -> DrawdownResult:

        if peak_equity <= 0:
            raise ValueError(
                "Peak Equity deve ser maior que zero."
            )

        current_equity = min(
            current_equity,
            peak_equity,
        )

        drawdown = (
            (peak_equity - current_equity)
            / peak_equity
        ) * 100

        approved = (
            drawdown < self.max_drawdown_percent
        )

        metadata = {

            "direction": setup.direction,

            "entry_strategy": setup.entry_strategy,

            "signal_quality": setup.signal_quality,

            "remaining_drawdown": max(
                self.max_drawdown_percent - drawdown,
                0.0,
            ),
        }

        if approved:

            return DrawdownResult(

                approved=True,

                reason="Drawdown within limit.",

                peak_equity=round(peak_equity, 2),

                current_equity=round(current_equity, 2),

                drawdown_percent=round(drawdown, 2),

                max_drawdown_percent=self.max_drawdown_percent,

                metadata=metadata,
            )

        return DrawdownResult(

            approved=False,

            reason="Maximum drawdown exceeded.",

            peak_equity=round(peak_equity, 2),

            current_equity=round(current_equity, 2),

            drawdown_percent=round(drawdown, 2),

            max_drawdown_percent=self.max_drawdown_percent,

            metadata=metadata,
        )