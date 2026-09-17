import pandas as pd

from backtest.anti_extension import AntiExtensionFilter


def _frame():
    return pd.DataFrame(
        {
            "decision": ["LONG", "LONG", "SHORT", "SHORT", "NO_TRADE"],
            "vwap_distance_atr": [2.5, 1.0, -2.5, -1.0, 9.0],
            "keltner_position": [0.8, 0.4, -0.8, -0.4, 9.0],
        }
    )


def test_directional_extension_is_symmetric():
    out = AntiExtensionFilter().enrich(_frame())
    assert out.loc[0, "directional_vwap_extension_atr"] == 2.5
    assert out.loc[2, "directional_vwap_extension_atr"] == 2.5
    assert out.loc[0, "directional_keltner_extension"] == 0.8
    assert out.loc[2, "directional_keltner_extension"] == 0.8


def test_combined_either_blocks_any_overextension():
    out = AntiExtensionFilter().apply(_frame(), "combined_either")
    assert out.loc[0, "decision"] == "NO_TRADE"
    assert out.loc[1, "decision"] == "LONG"
    assert out.loc[2, "decision"] == "NO_TRADE"
    assert out.loc[3, "decision"] == "SHORT"


def test_combined_both_requires_both_dimensions():
    frame = _frame()
    frame.loc[0, "keltner_position"] = 0.5
    out = AntiExtensionFilter().apply(frame, "combined_both")
    assert out.loc[0, "decision"] == "LONG"
    assert out.loc[2, "decision"] == "NO_TRADE"


def test_no_trade_rows_are_not_rejected():
    out = AntiExtensionFilter().apply(_frame(), "combined_either")
    assert bool(out.loc[4, "anti_extension_rejected"]) is False
