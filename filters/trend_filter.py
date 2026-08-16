from __future__ import annotations

from market.market_context import MarketContext

from filters.base_filter import BaseFilter
from filters.filter_result import FilterResult


class TrendFilter(BaseFilter):
    """
    Avalia a qualidade da tendência informada
    pelo MarketContext.

    Este filtro NÃO decide sozinho se uma operação
    será aprovada.

    Ele apenas produz uma pontuação padronizada.
    """

    name = "Trend Filter"

    def evaluate(
        self,
        context: MarketContext,
    ) -> FilterResult:

        trend = context.trend

        if trend == "STRONG_BULL":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction="LONG",
                score=100,
                grade="A+",
                reason="Strong Bull Trend",
                metadata={
                    "trend": trend
                },
            )

        if trend == "STRONG_BEAR":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction="SHORT",
                score=100,
                grade="A+",
                reason="Strong Bear Trend",
                metadata={
                    "trend": trend
                },
            )

        if trend == "WEAK_BULL":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction="LONG",
                score=82,
                grade="A",
                reason="Weak Bull Trend",
                metadata={
                    "trend": trend
                },
            )

        if trend == "WEAK_BEAR":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction="SHORT",
                score=82,
                grade="A",
                reason="Weak Bear Trend",
                metadata={
                    "trend": trend
                },
            )

        return FilterResult(
            approved=False,
            filter_name=self.name,
            direction="NONE",
            score=35,
            grade="C",
            reason="Range Market",
            metadata={
                "trend": trend
            },
        )