from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests


class BinanceDataVisionLoader:
    BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
    COLUMNS = [
        "timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]

    def __init__(self, session=None):
        self.session = session or requests.Session()

    def _daily_url(self, symbol: str, interval: str, day: date) -> str:
        stamp = day.isoformat()
        return f"{self.BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"

    def fetch_day(self, symbol: str, interval: str, day: date) -> pd.DataFrame:
        response = self.session.get(self._daily_url(symbol, interval, day), timeout=60)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            csv_name = archive.namelist()[0]
            frame = pd.read_csv(archive.open(csv_name), header=None, names=self.COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="us", utc=True, errors="coerce")
        # Older archives used milliseconds. Retry rows that became implausibly old/NaT.
        raw = pd.read_csv(BytesIO(ZipFile(BytesIO(response.content)).read(ZipFile(BytesIO(response.content)).namelist()[0])), header=None, names=self.COLUMNS)
        if frame["timestamp"].isna().all() or (not frame.empty and frame["timestamp"].dt.year.max() < 2020):
            frame = raw
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
        for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
        return frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    def fetch_window(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        frames = []
        day = start.date()
        while day <= end.date():
            frames.append(self.fetch_day(symbol, interval, day))
            day += timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=self.COLUMNS)
        result = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
        start_naive = start.astimezone(timezone.utc).replace(tzinfo=None)
        end_naive = end.astimezone(timezone.utc).replace(tzinfo=None)
        return result[(result["timestamp"] >= start_naive) & (result["timestamp"] < end_naive)].reset_index(drop=True)
