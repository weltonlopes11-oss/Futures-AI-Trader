from datetime import datetime, timezone

from backtest.binance_historical_collector import BinanceHistoricalDataCollector


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        return None
    def json(self):
        return self.payload


class FakeSession:
    def get(self, url, params, timeout):
        if "openInterestHist" in url:
            return FakeResponse([
                {"symbol": "ETHUSDT", "sumOpenInterest": "100", "timestamp": 1000},
                {"symbol": "ETHUSDT", "sumOpenInterest": "100", "timestamp": 2000},
            ])
        return FakeResponse([
            [1000, "10", "11", "9", "10.5", "20", 1999, "200", 5, "10", "100", "0"],
            [2000, "10.5", "12", "10", "11", "30", 2999, "300", 6, "15", "150", "0"],
        ])


def test_collector_preserves_equal_official_oi_observations():
    collector = BinanceHistoricalDataCollector(session=FakeSession(), pause_seconds=0)
    start = datetime.fromtimestamp(0, tz=timezone.utc)
    end = datetime.fromtimestamp(3, tz=timezone.utc)
    candles = collector.fetch_klines("ETHUSDT", "15m", start, end)
    oi = collector.fetch_open_interest("ETHUSDT", "15m", start, end)
    merged = collector.align_open_interest(candles, oi)

    assert len(oi) == 2
    assert len(merged) == 2
    assert merged.iloc[-1]["open_interest"] == 100
    assert merged.iloc[-1]["open_interest_change_pct"] == 0
    assert merged.iloc[-1]["open_interest_source_timestamp"] > merged.iloc[0]["open_interest_source_timestamp"]
