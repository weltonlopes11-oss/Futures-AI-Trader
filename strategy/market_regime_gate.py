from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Side = Literal["LONG", "SHORT"]


class GateState(str, Enum):
    ON = "ON"
    OFF = "OFF"


OFFICIAL_REGIMES_V1 = frozenset(
    {
        "LONG|LOW|CHOP",
        "SHORT|NORMAL|MIXED",
    }
)


@dataclass(frozen=True)
class RegimeSnapshot:
    side: Side
    regime_4h: str
    atr_price: float
    atr_price_q33: float
    atr_price_q67: float
    er20: float

    @property
    def volatility(self) -> str:
        if self.atr_price <= self.atr_price_q33:
            return "LOW"
        if self.atr_price <= self.atr_price_q67:
            return "NORMAL"
        return "HIGH"

    @property
    def efficiency(self) -> str:
        if self.er20 < 0.15:
            return "CHOP"
        if self.er20 < 0.30:
            return "MIXED"
        return "EFFICIENT"

    @property
    def regime_key(self) -> str:
        return f"{self.side}|{self.volatility}|{self.efficiency}"


@dataclass(frozen=True)
class RegimeDecision:
    state: GateState
    regime_key: str
    reason: str


def decide_regime(snapshot: RegimeSnapshot) -> RegimeDecision:
    required_4h = "BULLISH" if snapshot.side == "LONG" else "BEARISH"

    if snapshot.regime_4h != required_4h:
        return RegimeDecision(
            state=GateState.OFF,
            regime_key=snapshot.regime_key,
            reason=f"4H regime {snapshot.regime_4h} does not authorize {snapshot.side}",
        )

    if snapshot.regime_key not in OFFICIAL_REGIMES_V1:
        return RegimeDecision(
            state=GateState.OFF,
            regime_key=snapshot.regime_key,
            reason="Regime outside official robust set v1",
        )

    return RegimeDecision(
        state=GateState.ON,
        regime_key=snapshot.regime_key,
        reason="4H direction + adaptive volatility + ER20 authorize official trade",
    )
