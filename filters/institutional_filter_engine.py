from __future__ import annotations

from statistics import mean

from filters.base_filter import BaseFilter
from filters.filter_result import FilterResult
from filters.institutional_score import InstitutionalScore
from market.market_context import MarketContext


class InstitutionalFilterEngine:
    """
    Executa todos os filtros institucionais.
    """

    def __init__(
        self,
        filters: list[BaseFilter],
    ):
        self.filters = filters

    def evaluate(
        self,
        context: MarketContext,
    ) -> InstitutionalScore:

        results: list[FilterResult] = []

        for filter_instance in self.filters:
            result = filter_instance.evaluate(context)
            results.append(result)

        scores = [result.score for result in results]

        approvals = [result.approved for result in results]

        final_score = mean(scores) if scores else 0.0

        approved = all(approvals)

        return InstitutionalScore(
            approved=approved,
            score=round(final_score, 2),

            # Confiança consolidada (0.0–1.0)
            confidence=round(final_score / 100.0, 4),

            results=results,

            metadata={
                "filters": len(results),
                "approved_filters": sum(approvals),
                "rejected_filters": len(results) - sum(approvals),
            },
        )