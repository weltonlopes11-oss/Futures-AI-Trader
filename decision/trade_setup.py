from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from decision.signal_score import SignalScore
from decision.trade_decision import TradeDecision


EntryType = Literal[
    "MARKET",
    "LIMIT",
]

EntryStrategy = Literal[
    "BREAKOUT",
    "PULLBACK",
    "TREND_CONTINUATION",
]


@dataclass(slots=True)
class TradeSetup:
    """
    Define completamente COMO uma operação deverá
    ser executada.

    Não envia ordens.

    Não conhece Binance.

    Não calcula risco.

    Apenas descreve a estratégia operacional.
    """

    action: str

    direction: str

    entry_type: EntryType

    entry_strategy: EntryStrategy

    allow_scale_in: bool

    allow_scale_out: bool

    allow_pyramiding: bool

    signal_quality: str

    confidence: float

    metadata: dict[str, Any] = field(default_factory=dict)


class TradeSetupBuilder:
    """
    Constrói o TradeSetup a partir da decisão
    operacional e da qualidade do sinal.
    """

    def build(
        self,
        decision: TradeDecision,
        signal: SignalScore,
    ) -> TradeSetup:

        # ---------------------------------------
        # Estratégia de entrada
        # ---------------------------------------

        if signal.is_excellent:

            entry_type = "LIMIT"

            entry_strategy = "PULLBACK"

            scale_in = True

            pyramiding = True

        elif signal.is_good:

            entry_type = "MARKET"

            entry_strategy = "TREND_CONTINUATION"

            scale_in = False

            pyramiding = False

        elif signal.is_average:

            entry_type = "LIMIT"

            entry_strategy = "BREAKOUT"

            scale_in = False

            pyramiding = False

        else:

            entry_type = "MARKET"

            entry_strategy = "BREAKOUT"

            scale_in = False

            pyramiding = False

        return TradeSetup(

            action=decision.action,

            direction=decision.direction,

            entry_type=entry_type,

            entry_strategy=entry_strategy,

            allow_scale_in=scale_in,

            allow_scale_out=False,

            allow_pyramiding=pyramiding,

            signal_quality=signal.quality,

            confidence=signal.confidence,

            metadata={

                "recommendation": signal.recommendation,

                "institutional_score": decision.institutional_score,

                "decision_reason": decision.reason,
            },
        )