from __future__ import annotations

from market.market_context import MarketContext

from filters.base_filter import BaseFilter
from filters.filter_result import FilterResult


class VolatilityFilter(BaseFilter):
    """
    Avalia a volatilidade do mercado utilizando
    o MarketContext.

    Este filtro gera uma pontuação baseada na
    classificação da volatilidade.
    """

    name = "Volatility Filter"

    def evaluate(
        self,
        context: MarketContext,
    ) -> FilterResult:

        volatility = context.volatility

        detector = context.metadata.get("volatility")

        atr_percent = 0.0

        if detector is not None:
            atr_percent = detector.atr_percent

        # ---------------------------------------
        # Volatilidade normal
        # ---------------------------------------

        if volatility == "NORMAL":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction=context.direction,
                score=100,
                grade="A+",
                reason="Normal Volatility",
                metadata={
                    "volatility": volatility,
                    "atr_percent": atr_percent,
                },
            )

        # ---------------------------------------
        # Baixa volatilidade
        # ---------------------------------------

        if volatility == "LOW":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction=context.direction,
                score=80,
                grade="A",
                reason="Low Volatility",
                metadata={
                    "volatility": volatility,
                    "atr_percent": atr_percent,
                },
            )

        # ---------------------------------------
        # Alta volatilidade
        # ---------------------------------------

        return FilterResult(
            approved=False,
            filter_name=self.name,
            direction="NONE",
            score=40,
            grade="C",
            reason="High Volatility",
            metadata={
                "volatility": volatility,
                "atr_percent": atr_percent,
            },
        )