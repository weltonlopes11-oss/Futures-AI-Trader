from __future__ import annotations

from market.market_context import MarketContext

from filters.base_filter import BaseFilter
from filters.filter_result import FilterResult


class RegimeFilter(BaseFilter):
    """
    Avalia o regime consolidado do mercado.

    O RegimeDetector é responsável por identificar
    o regime. Este filtro apenas atribui uma
    pontuação para esse regime.
    """

    name = "Regime Filter"

    def evaluate(
        self,
        context: MarketContext,
    ) -> FilterResult:

        regime = context.regime

        # ----------------------------------------
        # STRONG BULL
        # ----------------------------------------

        if regime == "STRONG_BULL":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction="LONG",
                score=100,
                grade="A+",
                reason="Strong Bull Regime",
                metadata={
                    "regime": regime,
                    "direction": context.direction,
                },
            )

        # ----------------------------------------
        # STRONG BEAR
        # ----------------------------------------

        if regime == "STRONG_BEAR":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction="SHORT",
                score=100,
                grade="A+",
                reason="Strong Bear Regime",
                metadata={
                    "regime": regime,
                    "direction": context.direction,
                },
            )

        # ----------------------------------------
        # WEAK BULL
        # ----------------------------------------

        if regime == "WEAK_BULL":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction="LONG",
                score=85,
                grade="A",
                reason="Weak Bull Regime",
                metadata={
                    "regime": regime,
                    "direction": context.direction,
                },
            )

        # ----------------------------------------
        # WEAK BEAR
        # ----------------------------------------

        if regime == "WEAK_BEAR":

            return FilterResult(
                approved=True,
                filter_name=self.name,
                direction="SHORT",
                score=85,
                grade="A",
                reason="Weak Bear Regime",
                metadata={
                    "regime": regime,
                    "direction": context.direction,
                },
            )

        # ----------------------------------------
        # RANGE
        # ----------------------------------------

        if regime == "RANGE":

            return FilterResult(
                approved=False,
                filter_name=self.name,
                direction="NONE",
                score=35,
                grade="C",
                reason="Range Market",
                metadata={
                    "regime": regime,
                    "direction": context.direction,
                },
            )

        # ----------------------------------------
        # HIGH VOLATILITY
        # ----------------------------------------

        if regime == "HIGH_VOLATILITY":

            return FilterResult(
                approved=False,
                filter_name=self.name,
                direction="NONE",
                score=25,
                grade="C",
                reason="High Volatility Regime",
                metadata={
                    "regime": regime,
                    "direction": context.direction,
                },
            )

        # ----------------------------------------
        # Regime desconhecido
        # ----------------------------------------

        return FilterResult(
            approved=False,
            filter_name=self.name,
            direction="NONE",
            score=0,
            grade="C",
            reason="Unknown Market Regime",
            metadata={
                "regime": regime,
                "direction": context.direction,
            },
        )