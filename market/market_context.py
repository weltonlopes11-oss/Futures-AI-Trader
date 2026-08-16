from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MarketContext:
    """
    Representa o contexto atual do mercado.

    Este objeto é produzido pelo MarketContextEngine e
    consumido pelos filtros institucionais, AI Interpreter,
    Trade Decision, Replay e Optimizer.
    """

    regime: str
    confidence: float

    trend: str

    volatility: str

    direction: str

    score: float

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            "trend": self.trend,
            "volatility": self.volatility,
            "direction": self.direction,
            "score": self.score,
            "metadata": self.metadata,
        }

    @property
    def is_bull(self) -> bool:
        return self.direction == "LONG"

    @property
    def is_bear(self) -> bool:
        return self.direction == "SHORT"

    @property
    def is_range(self) -> bool:
        return self.regime == "RANGE"

    @property
    def is_high_volatility(self) -> bool:
        return self.volatility == "HIGH"

    @property
    def is_low_volatility(self) -> bool:
        return self.volatility == "LOW"

    def __repr__(self) -> str:
        return (
            f"MarketContext("
            f"regime='{self.regime}', "
            f"trend='{self.trend}', "
            f"volatility='{self.volatility}', "
            f"direction='{self.direction}', "
            f"confidence={self.confidence:.2f}, "
            f"score={self.score:.2f})"
        )