from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ==========================================================
# Mercado
# ==========================================================

@dataclass(slots=True)
class MarketSnapshot:

    regime: str

    direction: str

    trend: str

    volatility: str

    confidence: float

    score: float

    metadata: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# Filtro individual
# ==========================================================

@dataclass(slots=True)
class FilterSnapshot:

    name: str

    approved: bool

    score: float

    reason: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# Institutional Score
# ==========================================================

@dataclass(slots=True)
class InstitutionalSnapshot:

    approved: bool

    score: float

    confidence: float

    approved_filters: int

    rejected_filters: int

    metadata: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# Signal
# ==========================================================

@dataclass(slots=True)
class SignalSnapshot:

    recommendation: str

    quality: str

    score: float

    confidence: float

    metadata: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# Decision
# ==========================================================

@dataclass(slots=True)
class DecisionSnapshot:

    action: str

    approved: bool

    reason: str

    institutional_score: float

    confidence: float

    metadata: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# Risk
# ==========================================================

@dataclass(slots=True)
class RiskSnapshot:

    approved: bool

    risk_level: str

    confidence: float

    reasons: list[str]

    metadata: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# Candle completo
# ==========================================================

@dataclass(slots=True)
class TelemetryRecord:

    timestamp: Any

    symbol: str

    close_price: float

    market: MarketSnapshot

    filters: list[FilterSnapshot]

    institutional: InstitutionalSnapshot

    signal: SignalSnapshot

    decision: DecisionSnapshot

    risk: RiskSnapshot

    metadata: dict[str, Any] = field(default_factory=dict)