from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import pandas as pd
import requests


BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_DATA_VISION = "https://data.binance.vision/data/futures/um/daily/premiumIndexKlines"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


@dataclass(frozen=True)
class BinancePositioningConfig:
    period: str = "1h"
    timeout_seconds: int = 20


class BinancePositioningContextLoader:
    """Load causal Binance USD-M derivatives positioning context."""

    def __init__(self, config: BinancePositioningConfig | None = None):
        self.config = config or BinancePositioningConfig()

    @staticmethod
    def _utc_ms(value: datetime) -> int:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)

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

    def _get(self, path: str, params: dict) -> list:
        response = requests.get(
            f"{BINANCE_FAPI}{path}", params=params, timeout=self.config.timeout_seconds
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Binance payload for {path}")
        return payload

    @staticmethod
    def _snapshot_or_fetch(snapshot_path: Path | None, fetcher: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        if snapshot_path and snapshot_path.exists():
            frame = pd.read_csv(snapshot_path)
            if "timestamp" in frame:
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            if "premium_open_time" in frame:
                frame["premium_open_time"] = pd.to_datetime(frame["premium_open_time"], utc=True)
            return frame.sort_values("timestamp").reset_index(drop=True)
        return fetcher()

    def _fetch_premium_data_vision_day(self, symbol: str, interval: str, day) -> pd.DataFrame:
        stamp = day.isoformat()
        url = (
            f"{BINANCE_DATA_VISION}/{symbol}/{interval}/"
            f"{symbol}-{interval}-{stamp}.zip"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            raw = pd.read_csv(archive.open(archive.namelist()[0]), header=None)

        raw = raw.iloc[:, : len(KLINE_COLUMNS)].copy()
        raw.columns = KLINE_COLUMNS[: raw.shape[1]]
        open_time = self._parse_epoch(raw["open_time"])
        close_time = self._parse_epoch(raw["close_time"])
        out = pd.DataFrame(
            {
                "timestamp": close_time,
                "premium_open_time": open_time,
                "premium_open": pd.to_numeric(raw["open"], errors="coerce"),
                "premium_high": pd.to_numeric(raw["high"], errors="coerce"),
                "premium_low": pd.to_numeric(raw["low"], errors="coerce"),
                "premium_close": pd.to_numeric(raw["close"], errors="coerce"),
            }
        )
        return out.dropna(subset=["timestamp", "premium_close"])

    def fetch_premium_index(self, symbol: str, start: datetime, end: datetime, interval: str = "15m", snapshot_path: Path | None = None) -> pd.DataFrame:
        def fetch() -> pd.DataFrame:
            frames = []
            day = start.date()
            while day <= end.date():
                try:
                    day_frame = self._fetch_premium_data_vision_day(symbol, interval, day)
                    if not day_frame.empty:
                        frames.append(day_frame)
                except requests.HTTPError as exc:
                    if exc.response is None or exc.response.status_code != 404:
                        raise
                day += timedelta(days=1)

            if frames:
                result = pd.concat(frames, ignore_index=True)
                result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
                start_ts = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
                end_ts = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
                result = result[(result["timestamp"] >= start_ts) & (result["timestamp"] < end_ts)]
                if not result.empty:
                    return result.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

            rows = self._get(
                "/fapi/v1/premiumIndexKlines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": self._utc_ms(start),
                    "endTime": self._utc_ms(end) - 1,
                    "limit": 1500,
                },
            )
            parsed = []
            for row in rows:
                open_time = pd.to_datetime(int(row[0]), unit="ms", utc=True)
                close_time = pd.to_datetime(int(row[6]), unit="ms", utc=True)
                parsed.append(
                    {
                        "timestamp": close_time,
                        "premium_open_time": open_time,
                        "premium_open": float(row[1]),
                        "premium_high": float(row[2]),
                        "premium_low": float(row[3]),
                        "premium_close": float(row[4]),
                    }
                )
            return pd.DataFrame(parsed).sort_values("timestamp").reset_index(drop=True)

        return self._snapshot_or_fetch(snapshot_path, fetch)

    def _fetch_ratio(self, endpoint: str, symbol: str, start: datetime, end: datetime, prefix: str, snapshot_path: Path | None = None) -> pd.DataFrame:
        def fetch() -> pd.DataFrame:
            rows = self._get(
                endpoint,
                {
                    "symbol": symbol,
                    "period": self.config.period,
                    "startTime": self._utc_ms(start),
                    "endTime": self._utc_ms(end) - 1,
                    "limit": 500,
                },
            )
            parsed = []
            for row in rows:
                item = {
                    "timestamp": pd.to_datetime(int(row["timestamp"]), unit="ms", utc=True),
                    f"{prefix}_ratio": float(row["longShortRatio"]),
                }
                if "longAccount" in row:
                    item[f"{prefix}_long"] = float(row["longAccount"])
                    item[f"{prefix}_short"] = float(row["shortAccount"])
                else:
                    item[f"{prefix}_long"] = float(row["longPosition"])
                    item[f"{prefix}_short"] = float(row["shortPosition"])
                parsed.append(item)
            return pd.DataFrame(parsed).sort_values("timestamp").reset_index(drop=True)

        return self._snapshot_or_fetch(snapshot_path, fetch)

    def fetch_global_ratio(self, symbol: str, start: datetime, end: datetime, snapshot_path: Path | None = None) -> pd.DataFrame:
        return self._fetch_ratio("/futures/data/globalLongShortAccountRatio", symbol, start, end, "global_ls", snapshot_path)

    def fetch_top_account_ratio(self, symbol: str, start: datetime, end: datetime, snapshot_path: Path | None = None) -> pd.DataFrame:
        return self._fetch_ratio("/futures/data/topLongShortAccountRatio", symbol, start, end, "top_account_ls", snapshot_path)

    def fetch_top_position_ratio(self, symbol: str, start: datetime, end: datetime, snapshot_path: Path | None = None) -> pd.DataFrame:
        return self._fetch_ratio("/futures/data/topLongShortPositionRatio", symbol, start, end, "top_position_ls", snapshot_path)

    @staticmethod
    def align_causally(signal: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
        if context.empty:
            return signal.copy()
        right = context.copy()
        left = signal.copy()
        left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
        right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
        left = left.dropna(subset=["timestamp"]).sort_values("timestamp")
        right = right.dropna(subset=["timestamp"]).sort_values("timestamp")
        return pd.merge_asof(left, right, on="timestamp", direction="backward", allow_exact_matches=True)
