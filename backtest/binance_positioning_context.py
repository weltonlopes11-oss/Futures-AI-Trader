from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests


BINANCE_FAPI = "https://fapi.binance.com"


@dataclass(frozen=True)
class BinancePositioningConfig:
    period: str = "1h"
    timeout_seconds: int = 20


class BinancePositioningContextLoader:
    """Load causal Binance USD-M derivatives positioning context.

    Public statistics such as global/top-trader long-short ratios are only
    retained by Binance for a limited window. Every fetcher therefore accepts
    an optional deterministic CSV snapshot so research runs remain reproducible.
    """

    def __init__(self, config: BinancePositioningConfig | None = None):
        self.config = config or BinancePositioningConfig()

    @staticmethod
    def _utc_ms(value: datetime) -> int:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)

    def _get(self, path: str, params: dict) -> list:
        response = requests.get(
            f"{BINANCE_FAPI}{path}",
            params=params,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Binance payload for {path}")
        return payload

    @staticmethod
    def _snapshot_or_fetch(
        snapshot_path: Path | None,
        fetcher: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        if snapshot_path and snapshot_path.exists():
            frame = pd.read_csv(snapshot_path)
            if "timestamp" in frame:
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            if "premium_open_time" in frame:
                frame["premium_open_time"] = pd.to_datetime(frame["premium_open_time"], utc=True)
            return frame.sort_values("timestamp").reset_index(drop=True)
        return fetcher()

    def fetch_premium_index(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "15m",
        snapshot_path: Path | None = None,
    ) -> pd.DataFrame:
        def fetch() -> pd.DataFrame:
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
                        # premium_close is only knowable once this bar closes.
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

    def _fetch_ratio(
        self,
        endpoint: str,
        symbol: str,
        start: datetime,
        end: datetime,
        prefix: str,
        snapshot_path: Path | None = None,
    ) -> pd.DataFrame:
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
        return self._fetch_ratio(
            "/futures/data/globalLongShortAccountRatio",
            symbol,
            start,
            end,
            "global_ls",
            snapshot_path,
        )

    def fetch_top_account_ratio(self, symbol: str, start: datetime, end: datetime, snapshot_path: Path | None = None) -> pd.DataFrame:
        return self._fetch_ratio(
            "/futures/data/topLongShortAccountRatio",
            symbol,
            start,
            end,
            "top_account_ls",
            snapshot_path,
        )

    def fetch_top_position_ratio(self, symbol: str, start: datetime, end: datetime, snapshot_path: Path | None = None) -> pd.DataFrame:
        return self._fetch_ratio(
            "/futures/data/topLongShortPositionRatio",
            symbol,
            start,
            end,
            "top_position_ls",
            snapshot_path,
        )

    @staticmethod
    def align_causally(signal: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
        if context.empty:
            return signal.copy()
        right = context.copy().sort_values("timestamp")
        left = signal.copy().sort_values("timestamp")
        return pd.merge_asof(left, right, on="timestamp", direction="backward")
