from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from filters.filter_result import FilterResult


@dataclass(slots=True)
class InstitutionalScore:
    """
    Resultado consolidado da camada institucional.

    Este objeto representa a saída oficial do
    InstitutionalFilterEngine.
    """

    approved: bool

    score: float

    confidence: float

    results: list[FilterResult] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return not self.approved

    @property
    def total_filters(self) -> int:
        return len(self.results)

    @property
    def approved_filters(self) -> int:
        return sum(
            1
            for result in self.results
            if result.approved
        )

    @property
    def rejected_filters(self) -> int:
        return self.total_filters - self.approved_filters

    def get_filter(
        self,
        name: str,
    ) -> FilterResult | None:
        """
        Retorna um resultado de filtro pelo nome.
        """

        for result in self.results:
            if result.filter_name == name:
                return result

        return None

    def to_dict(self) -> dict[str, Any]:
        """
        Representação serializável.
        """

        return {
            "approved": self.approved,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 4),
            "total_filters": self.total_filters,
            "approved_filters": self.approved_filters,
            "rejected_filters": self.rejected_filters,
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            "InstitutionalScore("
            f"approved={self.approved}, "
            f"score={self.score:.2f}, "
            f"confidence={self.confidence:.4f}, "
            f"approved_filters={self.approved_filters}/{self.total_filters}"
            ")"
        )