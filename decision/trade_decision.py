from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from filters.institutional_score import InstitutionalScore


@dataclass(slots=True)
class TradeDecision:
    """
    Representa a decisão oficial da plataforma.

    Toda decisão operacional deve ser encapsulada
    nesta classe.

    Ela será consumida pelo:

    - Risk Manager
    - Position Sizer
    - Execution Engine
    - Replay
    - Optimizer
    """

    action: str

    approved: bool

    confidence: float

    institutional_score: float

    direction: str

    reason: str

    score: InstitutionalScore

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.action == "LONG"

    @property
    def is_short(self) -> bool:
        return self.action == "SHORT"

    @property
    def is_no_trade(self) -> bool:
        return self.action == "NO_TRADE"

    def to_dict(self) -> dict:

        return {

            "action": self.action,

            "approved": self.approved,

            "confidence": self.confidence,

            "institutional_score": self.institutional_score,

            "direction": self.direction,

            "reason": self.reason,

            "metadata": self.metadata,
        }