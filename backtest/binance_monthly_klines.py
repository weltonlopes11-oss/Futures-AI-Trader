from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests


class BinanceMonthlyKlineLoader:
    """Efficient official Binance Data Vision monthly kline loader."""

    BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
    COLUMNS = [
        "timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]

    def __init__(self, session=None):
        self.session = session or requests.Session()

    @staticmethod
    def _parse_epoch(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            return pd.to_datetime(numeric, utc=True, errors="coerce")
        magnitude = float(valid.abs().median())
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

    @staticmethod
    def _month_iter(start: datetime, end: datetime):
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            yield y, m
            m += 1
            if m == 13:
                y += 1
                m = 1

    def fetch_window(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        frames = []
        for y, m in self._month_iter(start, end):
            stamp = f"{y:04d}-{m:02d}"
            url = f"{self.BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"
            response = self.session.get(url, timeout=60)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            with ZipFile(BytesIO(response.content)) as archive:
                csv_name = archive.namelist()[0]
                frame = pd.read_csv(archive.open(csv_name), header=None, names=self.COLUMNS)
            frame["timestamp"] = self._parse_epoch(frame["timestamp"])
            frame["close_time"] = self._parse_epoch(frame["close_time"])
            for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
            frame["close_time"] = frame["close_time"].dt.tz_localize(None)
            frames.append(frame)
        if not frames:
            raise RuntimeError(f"No monthly Binance klines for {symbol} {interval}")
        result = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp", "open", "high", "low", "close"])
        result = result.drop_duplicates("timestamp").sort_values("timestamp")
        start_naive = start.astimezone(timezone.utc).replace(tzinfo=None)
        end_naive = end.astimezone(timezone.utc).replace(tzinfo=None)
        result = result[(result["timestamp"] >= start_naive) & (result["timestamp"] < end_naive)].reset_index(drop=True)
        if result.empty:
            raise RuntimeError(f"Monthly Binance klines empty for {symbol} {interval}")
        return result
