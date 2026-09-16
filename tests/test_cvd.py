from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from backtest.cvd import CumulativeVolumeDelta


def _frame(rows):
    return pd.DataFrame(rows, columns=["timestamp", "volume", "taker_buy_base"])


def test_cvd_delta_uses_taker_buy_minus_taker_sell():
    frame = _frame([
        ("2026-09-01 00:00:00", 10.0, 7.0),
        ("2026-09-01 00:15:00", 8.0, 3.0),
        ("2026-09-01 00:30:00", 5.0, 2.5),
    ])
    result = CumulativeVolumeDelta().enrich(frame)

    assert result["taker_sell_base"].tolist() == [3.0, 5.0, 2.5]
    assert result["cvd_delta"].tolist() == [4.0, -2.0, 0.0]
    assert result["cvd"].tolist() == [4.0, 2.0, 2.0]


def test_appending_future_candles_does_not_change_past_cvd_features():
    original = _frame([
        (pd.Timestamp("2026-09-01") + pd.Timedelta(minutes=15*i), 10.0 + (i % 3), 6.0 + ((i % 5) - 2) * 0.2)
        for i in range(40)
    ])
    future = _frame([
        (pd.Timestamp("2026-09-01") + pd.Timedelta(minutes=15*i), 10.0 + (i % 4), 4.5 + ((i % 7) - 3) * 0.15)
        for i in range(40, 60)
    ])

    engine = CumulativeVolumeDelta()
    base = engine.enrich(original).set_index("timestamp")
    extended = engine.enrich(pd.concat([original, future], ignore_index=True)).set_index("timestamp")

    for column in ("cvd_delta", "cvd", "cvd_signal"):
        pdt.assert_series_equal(base[column], extended.loc[base.index, column], check_names=True)


def test_cvd_rejects_impossible_taker_buy_volume():
    frame = _frame([("2026-09-01 00:00:00", 10.0, 11.0)])
    try:
        CumulativeVolumeDelta().enrich(frame)
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("Expected impossible taker-buy volume to fail")
