from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import requests


@dataclass(frozen=True)
class BinanceWindow:
    symbol: str
    start_time: datetime
    end_time: datetime
    signal_interval: str = "15m"
    oi_period: str = "15m"


class BinanceHistoricalDataCollector:
    """Deterministic public USD-M Futures market-data collector.

    All requests use an explicit UTC start/end window. No API key is needed.
    The returned signal dataframe contains 15m candles and causally aligned OI.
    1h and 4h candles are collected separately for structural context.
    """

    BASE_URL = "https://fapi.binance.com"
    KLINE_LIMIT = 1000
    OI_LIMIT = 500

    def __init__(self, session=None, pause_seconds: float = 0.05):
        self.session = session or requests.Session()
        self.pause_seconds = pause_seconds

    @staticmethod
    def _ms(value: datetime) -> int:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)

    def _get(self, path: str, params: dict):
        response = self.session.get(
            f"{self.BASE_URL}{path}", params=params, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def fetch_klines(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        start_ms = self._ms(start_time)
        end_ms = self._ms(end_time)
        rows = []
        cursor = start_ms

        while cursor < end_ms:
            batch = self._get(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": self.KLINE_LIMIT,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 1
            if next_cursor <= cursor:
                raise RuntimeError("Binance kline pagination did not advance")
            cursor = next_cursor
            if len(batch) < self.KLINE_LIMIT:
                break
            time.sleep(self.pause_seconds)

        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        for column in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    def fetch_open_interest(self, symbol: str, period: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        start_ms = self._ms(start_time)
        end_ms = self._ms(end_time)
        rows = []
        cursor = start_ms

        while cursor < end_ms:
            batch = self._get(
                "/futures/data/openInterestHist",
                {
                    "symbol": symbol,
                    "period": period,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": self.OI_LIMIT,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1]["timestamp"]) + 1
            if next_cursor <= cursor:
                raise RuntimeError("Binance OI pagination did not advance")
            cursor = next_cursor
            if len(batch) < self.OI_LIMIT:
                break
            time.sleep(self.pause_seconds)

        if not rows:
            return pd.DataFrame(columns=["open_interest_source_timestamp", "open_interest"])
        frame = pd.DataFrame(rows)
        frame["open_interest_source_timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        frame["open_interest"] = pd.to_numeric(frame["sumOpenInterest"], errors="coerce")
        return frame[["open_interest_source_timestamp", "open_interest"]].drop_duplicates("open_interest_source_timestamp").sort_values("open_interest_source_timestamp").reset_index(drop=True)

    @staticmethod
    def align_open_interest(candles: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
        if candles.empty:
            return candles.copy()
        result = candles.sort_values("timestamp").copy()
        if oi.empty:
            result["open_interest_source_timestamp"] = pd.NaT
            result["open_interest"] = float("nan")
            result["open_interest_change_pct"] = float("nan")
            return result
        official = oi.sort_values("open_interest_source_timestamp").copy()
        official["open_interest_change_pct"] = official["open_interest"].pct_change() * 100.0
        return pd.merge_asof(
            result,
            official,
            left_on="timestamp",
            right_on="open_interest_source_timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

    def collect(self, window: BinanceWindow) -> dict[str, pd.DataFrame]:
        signal = self.fetch_klines(window.symbol, window.signal_interval, window.start_time, window.end_time)
        structure_1h = self.fetch_klines(window.symbol, "1h", window.start_time, window.end_time)
        regime_4h = self.fetch_klines(window.symbol, "4h", window.start_time, window.end_time)
        oi = self.fetch_open_interest(window.symbol, window.oi_period, window.start_time, window.end_time)
        signal = self.align_open_interest(signal, oi)
        return {"signal_15m": signal, "structure_1h": structure_1h, "regime_4h": regime_4h, "open_interest": oi}
