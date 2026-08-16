from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decision.trade_setup import TradeSetup


@dataclass(slots=True)
class RiskDecision:
    """
    Resultado consolidado da avaliação de risco.

    O RiskManager nunca envia ordens.

    Apenas informa se a operação pode ou não
    prosseguir para a camada de execução.
    """

    approved: bool

    confidence: float

    risk_level: str

    reasons: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


class RiskManager:
    """
    Orquestrador da camada de risco.

    Nesta primeira versão apenas coordena
    os validadores de risco.

    Os módulos especializados serão adicionados
    progressivamente.
    """

    def __init__(
        self,
        validators: list | None = None,
    ):

        self.validators = validators or []

    def evaluate(
        self,
        setup: TradeSetup,
    ) -> RiskDecision:

        reasons: list[str] = []

        metadata: dict[str, Any] = {}

        approved = True

        for validator in self.validators:

            result = validator.validate(setup)

            metadata[
                validator.__class__.__name__
            ] = result

            validator_ok = getattr(
                result,
                "approved",
                True,
            )

            if not validator_ok:

                approved = False

                reason = getattr(
                    result,
                    "reason",
                    validator.__class__.__name__,
                )

                reasons.append(reason)

        if approved:

            reasons.append(
                "All risk validations approved."
            )

        return RiskDecision(

            approved=approved,

            confidence=setup.confidence,

            risk_level=(
                "NORMAL"
                if approved
                else "BLOCKED"
            ),

            reasons=reasons,

            metadata=metadata,
        )