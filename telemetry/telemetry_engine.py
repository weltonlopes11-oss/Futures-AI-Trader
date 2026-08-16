from __future__ import annotations

from telemetry.telemetry_models import (
    DecisionSnapshot,
    FilterSnapshot,
    InstitutionalSnapshot,
    MarketSnapshot,
    RiskSnapshot,
    SignalSnapshot,
    TelemetryRecord,
)


class TelemetryEngine:

    def __init__(self, symbol: str):
        self.symbol = symbol

    def capture(
        self,
        *,
        timestamp,
        close_price: float,
        context,
        institutional_score,
        signal,
        decision,
        risk,
    ) -> TelemetryRecord:

        # ======================================================
        # Mercado
        # ======================================================

        market = MarketSnapshot(
            regime=context.regime,
            direction=context.direction,
            trend=context.trend,
            volatility=context.volatility,
            confidence=context.confidence,
            score=context.score,
            metadata=getattr(context, "metadata", {}),
        )

        # ======================================================
        # Filtros
        # ======================================================

        filters = []

        approved = 0
        rejected = 0

        for result in getattr(institutional_score, "results", []):

            if result.approved:
                approved += 1
            else:
                rejected += 1

            filters.append(
                FilterSnapshot(
                    name=getattr(
                        result,
                        "filter_name",
                        result.__class__.__name__,
                    ),
                    approved=result.approved,
                    score=result.score,
                    reason=getattr(result, "reason", ""),
                    metadata=getattr(result, "metadata", {}),
                )
            )

        # ======================================================
        # Score Institucional
        # ======================================================

        institutional = InstitutionalSnapshot(
            approved=institutional_score.approved,
            score=institutional_score.score,
            confidence=institutional_score.confidence,
            approved_filters=approved,
            rejected_filters=rejected,
            metadata=getattr(
                institutional_score,
                "metadata",
                {},
            ),
        )

        # ======================================================
        # Signal
        # ======================================================

        signal_snapshot = SignalSnapshot(
             recommendation=signal.recommendation,
             quality=signal.quality,
             score=signal.value,
             confidence=signal.confidence,
             metadata=getattr(signal, "metadata", {}),
        )

        # ======================================================
        # Decision
        # ======================================================

        decision_snapshot = DecisionSnapshot(
            action=decision.action,
            approved=decision.approved,
            reason=decision.reason,
            institutional_score=getattr(
                decision,
                "institutional_score",
                institutional_score.score,
            ),
            confidence=decision.confidence,
            metadata=getattr(decision, "metadata", {}),
        )

        # ======================================================
        # Risk
        # ======================================================

        risk_snapshot = RiskSnapshot(
            approved=risk.approved,
            risk_level=risk.risk_level,
            confidence=risk.confidence,
            reasons=risk.reasons,
            metadata=getattr(risk, "metadata", {}),
        )

        # ======================================================
        # Registro completo
        # ======================================================

        return TelemetryRecord(
            timestamp=timestamp,
            symbol=self.symbol,
            close_price=close_price,
            market=market,
            filters=filters,
            institutional=institutional,
            signal=signal_snapshot,
            decision=decision_snapshot,
            risk=risk_snapshot,
            metadata={},
        )