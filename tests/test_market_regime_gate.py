from strategy.market_regime_gate import (
    FAVORABLE_REGIMES_2025,
    GateState,
    RegimeSnapshot,
    decide_regime,
)


def test_long_low_chop_is_on():
    snap = RegimeSnapshot(
        side="LONG",
        regime_4h="BULLISH",
        atr_price=0.006,
        atr_price_q33=0.007,
        atr_price_q67=0.011,
        er20=0.10,
    )
    decision = decide_regime(snap)
    assert decision.state == GateState.ON
    assert decision.regime_key == "LONG|LOW|CHOP"


def test_short_normal_mixed_is_on():
    snap = RegimeSnapshot(
        side="SHORT",
        regime_4h="BEARISH",
        atr_price=0.009,
        atr_price_q33=0.007,
        atr_price_q67=0.011,
        er20=0.22,
    )
    decision = decide_regime(snap)
    assert decision.state == GateState.ON
    assert decision.regime_key == "SHORT|NORMAL|MIXED"


def test_wrong_4h_direction_is_off():
    snap = RegimeSnapshot(
        side="LONG",
        regime_4h="BEARISH",
        atr_price=0.006,
        atr_price_q33=0.007,
        atr_price_q67=0.011,
        er20=0.10,
    )
    assert decide_regime(snap).state == GateState.OFF


def test_unfavorable_efficiency_is_off():
    snap = RegimeSnapshot(
        side="LONG",
        regime_4h="BULLISH",
        atr_price=0.006,
        atr_price_q33=0.007,
        atr_price_q67=0.011,
        er20=0.40,
    )
    assert decide_regime(snap).state == GateState.OFF


def test_frozen_set_is_exact():
    assert FAVORABLE_REGIMES_2025 == frozenset(
        {
            "LONG|HIGH|CHOP",
            "LONG|LOW|CHOP",
            "LONG|LOW|MIXED",
            "LONG|NORMAL|MIXED",
            "SHORT|NORMAL|MIXED",
        }
    )
