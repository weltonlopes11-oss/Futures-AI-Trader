from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BacktestMetrics:
    """
    Resultado consolidado de um backtest.

    Esta classe representa o contrato oficial entre o
    BacktestEngine e futuras camadas como:

    - Reports
    - Dashboard
    - Replay
    - Optimizer
    - Machine Learning
    """

    total_candles: int

    approved_trades: int

    rejected_trades: int

    long_trades: int

    short_trades: int

    no_trade: int

    approval_rate: float

    rejection_rate: float

    quality_distribution: dict[str, int] = field(default_factory=dict)

    market_regimes: dict[str, int] = field(default_factory=dict)

    market_directions: dict[str, int] = field(default_factory=dict)

    recommendations: dict[str, int] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_operations(self) -> int:
        return self.long_trades + self.short_trades

    @property
    def traded_percentage(self) -> float:

        if self.total_candles == 0:
            return 0.0

        return round(
            self.total_operations * 100 / self.total_candles,
            2,
        )

    @property
    def ignored_percentage(self) -> float:

        if self.total_candles == 0:
            return 0.0

        return round(
            self.no_trade * 100 / self.total_candles,
            2,
        )

    def to_dict(self) -> dict:

        return {

            "total_candles": self.total_candles,

            "approved_trades": self.approved_trades,

            "rejected_trades": self.rejected_trades,

            "long_trades": self.long_trades,

            "short_trades": self.short_trades,

            "no_trade": self.no_trade,

            "approval_rate": self.approval_rate,

            "rejection_rate": self.rejection_rate,

            "traded_percentage": self.traded_percentage,

            "ignored_percentage": self.ignored_percentage,

            "quality_distribution": self.quality_distribution,

            "market_regimes": self.market_regimes,

            "market_directions": self.market_directions,

            "recommendations": self.recommendations,

            "metadata": self.metadata,
        }