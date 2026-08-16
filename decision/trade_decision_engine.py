from __future__ import annotations

from decision.trade_decision import TradeDecision
from filters.institutional_score import InstitutionalScore
from market.market_context import MarketContext


class TradeDecisionEngine:
    """
    Responsável pela decisão final de entrada.

    Este é o único componente autorizado a transformar
    uma análise institucional em uma decisão operacional.
    """

    def __init__(
        self,
        min_score: float = 80.0,
        min_confidence: float = 0.75,
    ):

        self.min_score = min_score
        self.min_confidence = min_confidence

    def evaluate(
        self,
        context: MarketContext,
        score: InstitutionalScore,
    ) -> TradeDecision:

        # -----------------------------
        # Operação reprovada
        # -----------------------------

        if not score.approved:

            return TradeDecision(
                action="NO_TRADE",
                approved=False,
                confidence=score.confidence,
                institutional_score=score.score,
                direction=context.direction,
                reason="Institutional filters rejected",
                score=score,
                metadata={
                    "market_regime": context.regime,
                    "trend": context.trend,
                    "volatility": context.volatility,
                },
            )

        # -----------------------------
        # Score insuficiente
        # -----------------------------

        if score.score < self.min_score:

            return TradeDecision(
                action="NO_TRADE",
                approved=False,
                confidence=score.confidence,
                institutional_score=score.score,
                direction=context.direction,
                reason="Institutional score below minimum",
                score=score,
                metadata={
                    "required_score": self.min_score,
                },
            )

        # -----------------------------
        # Confiança insuficiente
        # -----------------------------

        if score.confidence < self.min_confidence:

            return TradeDecision(
                action="NO_TRADE",
                approved=False,
                confidence=score.confidence,
                institutional_score=score.score,
                direction=context.direction,
                reason="Confidence below minimum",
                score=score,
                metadata={
                    "required_confidence": self.min_confidence,
                },
            )

        # -----------------------------
        # LONG
        # -----------------------------

        if context.direction == "LONG":

            return TradeDecision(
                action="LONG",
                approved=True,
                confidence=score.confidence,
                institutional_score=score.score,
                direction="LONG",
                reason="Institutional approval",
                score=score,
                metadata={
                    "market_regime": context.regime,
                },
            )

        # -----------------------------
        # SHORT
        # -----------------------------

        if context.direction == "SHORT":

            return TradeDecision(
                action="SHORT",
                approved=True,
                confidence=score.confidence,
                institutional_score=score.score,
                direction="SHORT",
                reason="Institutional approval",
                score=score,
                metadata={
                    "market_regime": context.regime,
                },
            )

        # -----------------------------
        # Sem direção válida
        # -----------------------------

        return TradeDecision(
            action="NO_TRADE",
            approved=False,
            confidence=score.confidence,
            institutional_score=score.score,
            direction="NEUTRAL",
            reason="No valid market direction",
            score=score,
            metadata={
                "market_regime": context.regime,
            },
        )