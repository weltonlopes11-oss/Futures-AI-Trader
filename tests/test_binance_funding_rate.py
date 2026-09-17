from datetime import datetime, timezone

import pandas as pd

from backtest.binance_funding_rate import BinanceFundingRateLoader


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return FakeResponse(self.payload)


def test_funding_loader_normalizes_and_filters_window():
    payload = [
        {"symbol": "ETHUSDT", "fundingTime": 1788220800005, "fundingRate": "0.00005716", "markPrice": "2466.63"},
        {"symbol": "ETHUSDT", "fundingTime": 1788249600000, "fundingRate": "-0.00001000", "markPrice": "2471.44"},
    ]
    session = FakeSession(payload)
    loader = BinanceFundingRateLoader(session=session)
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)

    frame = loader.fetch_window("ETHUSDT", start, end)

    assert len(frame) == 2
    assert str(frame["funding_source_timestamp"].dtype) == "datetime64[ns]"
    assert frame["funding_rate"].tolist() == [0.00005716, -0.00001]
    assert session.calls[0][1]["symbol"] == "ETHUSDT"


def test_funding_alignment_is_backward_only():
    candles = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-09-01 07:59:59",
            "2026-09-01 08:00:00",
            "2026-09-01 12:00:00",
        ]),
        "close": [1.0, 2.0, 3.0],
    })
    funding = pd.DataFrame({
        "funding_source_timestamp": pd.to_datetime([
            "2026-09-01 00:00:00",
            "2026-09-01 08:00:00",
        ]),
        "funding_rate": [0.0001, -0.0002],
        "funding_mark_price": [1.0, 2.0],
    })

    aligned = BinanceFundingRateLoader.align_causally(candles, funding)

    assert aligned["funding_rate"].tolist() == [0.0001, -0.0002, -0.0002]
    assert (aligned["funding_source_timestamp"] <= aligned["timestamp"]).all()
