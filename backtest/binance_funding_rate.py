from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests


class BinanceFundingRateLoader:
    """Historical USD-M perpetual funding rates from Binance public REST API."""

    BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
    LIMIT = 1000

    def __init__(self, session=None):
        self.session = session or requests.Session()

    @staticmethod
    def _to_utc_ms(value: datetime) -> int:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.astimezone(timezone.utc).timestamp() * 1000)

    def fetch_window(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end <= start:
            raise ValueError("end must be after start")

        start_ms = self._to_utc_ms(start)
        end_ms = self._to_utc_ms(end)
        cursor = start_ms
        rows: list[dict] = []

        while cursor < end_ms:
            response = self.session.get(
                self.BASE_URL,
                params={
                    "symbol": symbol,
                    "startTime": cursor,
                    "endTime": end_ms - 1,
                    "limit": self.LIMIT,
                },
                timeout=60,
            )
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                raise RuntimeError("Unexpected Binance funding-rate response")
            if not batch:
                break

            rows.extend(batch)
            last_time = max(int(item["fundingTime"]) for item in batch)
            next_cursor = last_time + 1
            if next_cursor <= cursor:
                raise RuntimeError("Funding-rate pagination did not advance")
            cursor = next_cursor

            if len(batch) < self.LIMIT:
                break

        if not rows:
            raise RuntimeError(
                f"Binance returned no funding rates for {symbol} between "
                f"{start.isoformat()} and {end.isoformat()}"
            )

        frame = pd.DataFrame(rows)
        required = {"fundingTime", "fundingRate"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Funding-rate response missing columns: {sorted(missing)}")

        frame["funding_source_timestamp"] = (
            pd.to_datetime(pd.to_numeric(frame["fundingTime"], errors="coerce"), unit="ms", utc=True, errors="coerce")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        if "markPrice" in frame.columns:
            frame["funding_mark_price"] = pd.to_numeric(frame["markPrice"], errors="coerce")
        else:
            frame["funding_mark_price"] = pd.NA

        start_naive = start.astimezone(timezone.utc).replace(tzinfo=None)
        end_naive = end.astimezone(timezone.utc).replace(tzinfo=None)
        frame = frame[
            (frame["funding_source_timestamp"] >= start_naive)
            & (frame["funding_source_timestamp"] < end_naive)
        ]
        frame = (
            frame[["funding_source_timestamp", "funding_rate", "funding_mark_price"]]
            .dropna(subset=["funding_source_timestamp", "funding_rate"])
            .drop_duplicates("funding_source_timestamp")
            .sort_values("funding_source_timestamp")
            .reset_index(drop=True)
        )
        if frame.empty:
            raise RuntimeError("Funding-rate window is empty after normalization")
        return frame

    @staticmethod
    def align_causally(candles: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
        left = candles.copy()
        right = funding.copy()
        left["timestamp"] = (
            pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        right["funding_source_timestamp"] = (
            pd.to_datetime(right["funding_source_timestamp"], utc=True, errors="coerce")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        left = left.dropna(subset=["timestamp"]).sort_values("timestamp")
        right = right.dropna(subset=["funding_source_timestamp"]).sort_values("funding_source_timestamp")
        return pd.merge_asof(
            left,
            right,
            left_on="timestamp",
            right_on="funding_source_timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
