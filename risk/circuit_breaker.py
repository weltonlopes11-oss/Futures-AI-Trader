from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class CircuitBreakerStatus:
    """
    Estado atual do Circuit Breaker.
    """

    state: str

    active: bool

    reason: str

    triggered_at: datetime | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """
    Responsável por interromper operações quando
    ocorrerem condições críticas.

    Estados possíveis:

    CLOSED
    WARNING
    OPEN
    HALTED
    """

    VALID_STATES = {
        "CLOSED",
        "WARNING",
        "OPEN",
        "HALTED",
    }

    def __init__(self):

        self._status = CircuitBreakerStatus(

            state="CLOSED",

            active=False,

            reason="System operating normally.",
        )

    @property
    def status(self) -> CircuitBreakerStatus:
        return self._status

    @property
    def is_closed(self) -> bool:
        return self._status.state == "CLOSED"

    @property
    def is_warning(self) -> bool:
        return self._status.state == "WARNING"

    @property
    def is_open(self) -> bool:
        return self._status.state == "OPEN"

    @property
    def is_halted(self) -> bool:
        return self._status.state == "HALTED"

    @property
    def can_trade(self) -> bool:
        return self._status.state == "CLOSED"

    def trigger(
        self,
        state: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> CircuitBreakerStatus:

        state = state.upper()

        if state not in self.VALID_STATES:

            raise ValueError(
                f"Estado inválido: {state}"
            )

        self._status = CircuitBreakerStatus(

            state=state,

            active=state != "CLOSED",

            reason=reason,

            triggered_at=datetime.utcnow(),

            metadata=metadata or {},
        )

        return self._status

    def reset(self) -> CircuitBreakerStatus:

        self._status = CircuitBreakerStatus(

            state="CLOSED",

            active=False,

            reason="Circuit Breaker reset.",

            triggered_at=None,

            metadata={},
        )

        return self._status