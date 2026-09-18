from strategy.market_regime_gate import (
    DISCOVERED_REGIMES_2025,
    ROBUST_REGIMES_2025_2026,
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


def test_discovered_but_not_robust_regime_is_off():
    snap = RegimeSnapshot(
        side="LONG",
        regime_4h="BULLISH",
        atr_price=0.013,
        atr_price_q33=0.007,
        atr_price_q67=0.011,
        er20=0.10,
    )
    assert snap.regime_key == "LONG|HIGH|CHOP"
    assert snap.regime_key in DISCOVERED_REGIMES_2025
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


def test_robust_set_is_exact():
    assert ROBUST_REGIMES_2025_2026 == frozenset(
        {
            "LONG|LOW|CHOP",
            "SHORT|NORMAL|MIXED",
        }
    )
