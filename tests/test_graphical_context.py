from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.graphical_context import GraphicalContext


def sample_frame(rows: int = 160) -> pd.DataFrame:
    ts = pd.date_range("2026-09-01", periods=rows, freq="15min", tz="UTC")
    base = 3000.0 + np.arange(rows, dtype=float) * 0.8
    wiggle = np.sin(np.arange(rows) / 5.0) * 4.0
    close = base + wiggle
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.5,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
            "volume": 100.0 + (np.arange(rows) % 17),
        }
    )


def test_graphical_context_has_expected_columns():
    out = GraphicalContext().enrich(sample_frame())
    expected = {
        "keltner_middle",
        "keltner_upper",
        "keltner_lower",
        "keltner_position",
        "adx",
        "plus_di",
        "minus_di",
        "session_vwap",
        "vwap_distance_atr",
        "prior_structure_high",
        "prior_structure_low",
    }
    assert expected.issubset(out.columns)


def test_future_change_does_not_modify_prior_features():
    frame = sample_frame()
    original = GraphicalContext().enrich(frame)
    changed = frame.copy()
    changed.loc[len(changed) - 1, ["high", "low", "close", "volume"]] = [5000, 1000, 4500, 999999]
    modified = GraphicalContext().enrich(changed)

    cols = [
        "keltner_middle",
        "keltner_upper",
        "keltner_lower",
        "adx",
        "plus_di",
        "minus_di",
        "session_vwap",
        "prior_structure_high",
        "prior_structure_low",
    ]
    pd.testing.assert_frame_equal(
        original.loc[:-2, cols].reset_index(drop=True),
        modified.loc[:-2, cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_market_structure_excludes_current_bar():
    frame = sample_frame(40)
    out = GraphicalContext().enrich(frame)
    i = 30
    expected_high = frame.loc[i - 20 : i - 1, "high"].max()
    expected_low = frame.loc[i - 20 : i - 1, "low"].min()
    assert out.loc[i, "prior_structure_high"] == expected_high
    assert out.loc[i, "prior_structure_low"] == expected_low


def test_vwap_resets_each_utc_day():
    frame = sample_frame(120)
    out = GraphicalContext().enrich(frame)
    ts = pd.to_datetime(out["timestamp"], utc=True)
    second_day_start = ts.dt.floor("D").ne(ts.dt.floor("D").shift(1))
    indices = list(out.index[second_day_start])
    assert len(indices) >= 2
    i = indices[1]
    typical = (out.loc[i, "high"] + out.loc[i, "low"] + out.loc[i, "close"]) / 3.0
    assert abs(out.loc[i, "session_vwap"] - typical) < 1e-9
