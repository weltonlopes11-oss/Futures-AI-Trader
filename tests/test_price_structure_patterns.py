from __future__ import annotations

import pandas as pd

from backtest.price_structure_patterns import PriceStructureConfig, PriceStructurePatterns


def _frame(highs, lows, closes=None, atr=2.0):
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-09-01", periods=len(highs), freq="15min", tz="UTC"),
            "high": highs,
            "low": lows,
            "close": closes,
            "atr": [atr] * len(highs),
        }
    )


def test_pivot_is_only_available_after_right_side_confirmation():
    f = _frame([1, 2, 5, 3, 2, 1, 2], [0, 0.5, 1, 0.8, 0.7, 0.6, 0.8])
    out = PriceStructurePatterns(PriceStructureConfig(pivot_span=2)).enrich(f)
    # Candidate high at index 2 can only be confirmed at index 4.
    assert not out.loc[2, "pivot_high_confirmed"]
    assert bool(out.loc[4, "pivot_high_confirmed"])
    assert out.loc[4, "confirmed_pivot_high_index"] == 2
    assert out.loc[4, "confirmed_pivot_high_price"] == 5


def test_future_changes_do_not_change_already_confirmed_pivot_state():
    base = _frame([1, 2, 5, 3, 2, 1, 2, 3], [0, 0.5, 1, 0.8, 0.7, 0.6, 0.8, 1])
    alt = base.copy()
    alt.loc[7, ["high", "low", "close"]] = [100, -100, 50]
    eng = PriceStructurePatterns(PriceStructureConfig(pivot_span=2))
    a = eng.enrich(base)
    b = eng.enrich(alt)
    cols = ["pivot_high_confirmed", "confirmed_pivot_high_price", "last_swing_high"]
    pd.testing.assert_frame_equal(a.loc[:6, cols], b.loc[:6, cols])


def test_hh_and_hl_classification():
    highs = [1,2,5,3,2,4,3,6,4,3,5,4,7,5,4]
    lows  = [0,1,2,1,0.5,1.5,1,2,1.5,1.2,2,1.8,3,2.5,2]
    out = PriceStructurePatterns(PriceStructureConfig(pivot_span=2)).enrich(_frame(highs, lows))
    assert "HH" in set(out["swing_high_class"])
    assert "HL" in set(out["swing_low_class"])


def test_double_top_uses_atr_tolerance_and_minimum_separation():
    highs = [1,2,5,3,2,1,2,4.8,3,2,1]
    lows = [0,0.5,1,0.8,0.6,0.4,0.7,1,0.8,0.6,0.5]
    out = PriceStructurePatterns(
        PriceStructureConfig(pivot_span=2, double_tolerance_atr=0.25, double_min_separation_bars=4)
    ).enrich(_frame(highs, lows, atr=1.0))
    assert bool(out["double_top"].any())


def test_bos_uses_previously_confirmed_level():
    highs = [1,2,5,3,2,2.5,3,3.5,6,6.5]
    lows = [0,0.5,1,0.8,0.7,1,1.2,1.5,2,2.5]
    closes = [0.5,1.5,4,2.5,2,2.3,2.8,3.4,5.5,6.2]
    out = PriceStructurePatterns(PriceStructureConfig(pivot_span=2)).enrich(_frame(highs, lows, closes))
    assert bool(out["bos_up"].any())
