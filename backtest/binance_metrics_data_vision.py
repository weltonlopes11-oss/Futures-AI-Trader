from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests


class BinanceMetricsDataVisionLoader:
    BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"

    def __init__(self, session=None):
        self.session = session or requests.Session()

    def _daily_url(self, symbol: str, day: date) -> str:
        return f"{self.BASE_URL}/{symbol}/{symbol}-metrics-{day.isoformat()}.zip"

    def fetch_day(self, symbol: str, day: date) -> pd.DataFrame:
        response = self.session.get(self._daily_url(symbol, day), timeout=60)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            frame = pd.read_csv(archive.open(archive.namelist()[0]))
        required = {"create_time", "sum_open_interest"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Binance metrics archive missing columns: {sorted(missing)}")
        frame["open_interest_source_timestamp"] = pd.to_datetime(frame["create_time"], utc=True, errors="coerce").dt.tz_localize(None)
        frame["open_interest"] = pd.to_numeric(frame["sum_open_interest"], errors="coerce")
        return frame[["open_interest_source_timestamp", "open_interest"]].dropna().drop_duplicates("open_interest_source_timestamp").sort_values("open_interest_source_timestamp").reset_index(drop=True)

    def fetch_window(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        frames = []
        day = start.date()
        while day <= end.date():
            frames.append(self.fetch_day(symbol, day))
            day += timedelta(days=1)
        result = pd.concat(frames, ignore_index=True).drop_duplicates("open_interest_source_timestamp").sort_values("open_interest_source_timestamp")
        start_naive = start.astimezone(timezone.utc).replace(tzinfo=None)
        end_naive = end.astimezone(timezone.utc).replace(tzinfo=None)
        result = result[(result["open_interest_source_timestamp"] >= start_naive) & (result["open_interest_source_timestamp"] < end_naive)].reset_index(drop=True)
        if result.empty:
            raise RuntimeError("Binance Data Vision returned no Open Interest metrics")
        result["open_interest_change_pct"] = result["open_interest"].pct_change() * 100.0
        return result

    @staticmethod
    def align_causally(candles: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
        candles_aligned = candles.copy()
        oi_aligned = oi.copy()

        # pandas.merge_asof requires both merge keys to have exactly the same
        # datetime dtype/resolution. Binance candle and Data Vision timestamps
        # can arrive as datetime64[ms] and datetime64[us], respectively.
        candles_aligned["timestamp"] = (
            pd.to_datetime(candles_aligned["timestamp"], utc=True, errors="coerce")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        oi_aligned["open_interest_source_timestamp"] = (
            pd.to_datetime(oi_aligned["open_interest_source_timestamp"], utc=True, errors="coerce")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )

        candles_aligned = candles_aligned.dropna(subset=["timestamp"]).sort_values("timestamp")
        oi_aligned = oi_aligned.dropna(subset=["open_interest_source_timestamp"]).sort_values("open_interest_source_timestamp")

        return pd.merge_asof(
            candles_aligned,
            oi_aligned,
            left_on="timestamp",
            right_on="open_interest_source_timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
