from __future__ import annotations

import pandas as pd


def add_entry_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a causal, non-fitted entry confirmation score.

    The frozen Keltner confirmation remains mandatory. Three optional graphical
    confirmations contribute one point each:
      - DMI direction agrees with the trade direction;
      - ADX is rising;
      - price is on the correct side of session VWAP.

    A natural anti-chase guard rejects entries already beyond the outer Keltner
    band (|keltner_position| > 1). No PnL-derived thresholds are used.
    """
    required = {
        "decision",
        "keltner_long_confirm",
        "keltner_short_confirm",
        "plus_di",
        "minus_di",
        "adx_rising",
        "vwap_long_confirm",
        "vwap_short_confirm",
        "keltner_position",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"entry quality missing columns: {sorted(missing)}")

    c = frame.copy()
    is_long = c["decision"].astype(str).str.upper().eq("LONG")
    is_short = c["decision"].astype(str).str.upper().eq("SHORT")

    dmi_agree = (is_long & (c["plus_di"] > c["minus_di"])) | (
        is_short & (c["minus_di"] > c["plus_di"])
    )
    vwap_agree = (is_long & c["vwap_long_confirm"]) | (
        is_short & c["vwap_short_confirm"]
    )
    keltner_confirm = (is_long & c["keltner_long_confirm"]) | (
        is_short & c["keltner_short_confirm"]
    )

    c["entry_quality_score"] = (
        dmi_agree.astype(int)
        + c["adx_rising"].fillna(False).astype(int)
        + vwap_agree.astype(int)
    )
    c["entry_not_overextended"] = c["keltner_position"].abs() <= 1.0
    c["entry_quality_pass"] = (
        keltner_confirm
        & c["entry_not_overextended"].fillna(False)
        & (c["entry_quality_score"] >= 2)
    )
    return c


def apply_entry_quality(frame: pd.DataFrame) -> pd.DataFrame:
    c = add_entry_quality(frame)
    original = c["decision"].astype(str).str.upper()
    c["decision"] = "NO_TRADE"
    keep = original.isin(["LONG", "SHORT"]) & c["entry_quality_pass"]
    c.loc[keep, "decision"] = original.loc[keep]
    return c
