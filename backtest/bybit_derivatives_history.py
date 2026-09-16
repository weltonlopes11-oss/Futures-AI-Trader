from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


class BybitDerivativesHistoryLoader:
    """Historical public Bybit linear-perpetual OI and funding context.

    The loader is read-only, uses official V5 public market endpoints, preserves
    source timestamps, and exposes causal backward alignment helpers for the
    15-minute Binance signal frame.
    """

    BASE_URL = "https://api.bybit.com"
    OI_PATH = "/v5/market/open-interest"
    FUNDING_PATH = "/v5/market/funding/history"
    LIMIT = 200

    def __init__(self, session=None, timeout_seconds: int = 30):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _utc_ms(value: datetime) -> int:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.astimezone(timezone.utc).timestamp() * 1000)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(
            f"{self.BASE_URL}{path}", params=params, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        payload = response.json()
        if int(payload.get("retCode", -1)) != 0:
            raise RuntimeError(
                f"Bybit {path} failed: retCode={payload.get('retCode')} "
                f"retMsg={payload.get('retMsg')}"
            )
        return payload

    @staticmethod
    def _normalize_oi(frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        required = {"timestamp", "openInterest"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Bybit OI data missing columns: {sorted(missing)}")

        out = pd.DataFrame(
            {
                "bybit_oi_source_timestamp": pd.to_datetime(
                    pd.to_numeric(frame["timestamp"], errors="coerce"),
                    unit="ms",
                    utc=True,
                    errors="coerce",
                ),
                "bybit_open_interest": pd.to_numeric(frame["openInterest"], errors="coerce"),
            }
        ).dropna(subset=["bybit_oi_source_timestamp", "bybit_open_interest"])

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")

        out = out[
            (out["bybit_oi_source_timestamp"] >= start_ts)
            & (out["bybit_oi_source_timestamp"] < end_ts)
        ]
        out = (
            out.drop_duplicates("bybit_oi_source_timestamp")
            .sort_values("bybit_oi_source_timestamp")
            .reset_index(drop=True)
        )
        if out.empty:
            raise RuntimeError("Bybit OI window is empty after normalization")
        out["bybit_open_interest_change_pct"] = out["bybit_open_interest"].pct_change()
        return out

    @staticmethod
    def _normalize_funding(frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        required = {"fundingRateTimestamp", "fundingRate"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Bybit funding data missing columns: {sorted(missing)}")

        out = pd.DataFrame(
            {
                "bybit_funding_source_timestamp": pd.to_datetime(
                    pd.to_numeric(frame["fundingRateTimestamp"], errors="coerce"),
                    unit="ms",
                    utc=True,
                    errors="coerce",
                ),
                "bybit_funding_rate": pd.to_numeric(frame["fundingRate"], errors="coerce"),
            }
        ).dropna(subset=["bybit_funding_source_timestamp", "bybit_funding_rate"])

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")

        out = out[
            (out["bybit_funding_source_timestamp"] >= start_ts)
            & (out["bybit_funding_source_timestamp"] < end_ts)
        ]
        out = (
            out.drop_duplicates("bybit_funding_source_timestamp")
            .sort_values("bybit_funding_source_timestamp")
            .reset_index(drop=True)
        )
        if out.empty:
            raise RuntimeError("Bybit funding window is empty after normalization")
        return out

    def fetch_open_interest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "15min",
        snapshot_path: str | Path | None = None,
    ) -> pd.DataFrame:
        if snapshot_path is not None and Path(snapshot_path).exists():
            return self._normalize_oi(pd.read_csv(snapshot_path), start, end)

        params: dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": interval,
            "startTime": self._utc_ms(start),
            "endTime": self._utc_ms(end) - 1,
            "limit": self.LIMIT,
        }
        rows: list[dict[str, Any]] = []
        seen_cursors: set[str] = set()

        while True:
            payload = self._get(self.OI_PATH, params)
            result = payload.get("result") or {}
            batch = result.get("list") or []
            rows.extend(batch)
            cursor = str(result.get("nextPageCursor") or "")
            if not cursor or cursor in seen_cursors or len(batch) < self.LIMIT:
                break
            seen_cursors.add(cursor)
            params["cursor"] = cursor

        if not rows:
            raise RuntimeError("Bybit returned no historical open interest")
        return self._normalize_oi(pd.DataFrame(rows), start, end)

    def fetch_funding(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        snapshot_path: str | Path | None = None,
    ) -> pd.DataFrame:
        if snapshot_path is not None and Path(snapshot_path).exists():
            return self._normalize_funding(pd.read_csv(snapshot_path), start, end)

        end_ms = self._utc_ms(end) - 1
        start_ms = self._utc_ms(start)
        rows: list[dict[str, Any]] = []
        cursor_end = end_ms

        # Funding observations are sparse (typically every 8h for ETHUSDT).
        # Paginate backwards deterministically because the public endpoint has
        # no cursor and returns up to 200 rows ending at endTime.
        while cursor_end >= start_ms:
            payload = self._get(
                self.FUNDING_PATH,
                {
                    "category": "linear",
                    "symbol": symbol,
                    "startTime": start_ms,
                    "endTime": cursor_end,
                    "limit": self.LIMIT,
                },
            )
            result = payload.get("result") or {}
            batch = result.get("list") or []
            if not batch:
                break
            rows.extend(batch)
            oldest = min(int(item["fundingRateTimestamp"]) for item in batch)
            next_end = oldest - 1
            if next_end >= cursor_end or len(batch) < self.LIMIT:
                break
            cursor_end = next_end

        if not rows:
            raise RuntimeError("Bybit returned no historical funding rates")
        return self._normalize_funding(pd.DataFrame(rows), start, end)

    @staticmethod
    def align_oi_causally(candles: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
        left = candles.copy()
        right = oi.copy()
        left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
        right["bybit_oi_source_timestamp"] = pd.to_datetime(
            right["bybit_oi_source_timestamp"], utc=True, errors="coerce"
        )
        return pd.merge_asof(
            left.dropna(subset=["timestamp"]).sort_values("timestamp"),
            right.dropna(subset=["bybit_oi_source_timestamp"]).sort_values("bybit_oi_source_timestamp"),
            left_on="timestamp",
            right_on="bybit_oi_source_timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

    @staticmethod
    def align_funding_causally(candles: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
        left = candles.copy()
        right = funding.copy()
        left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
        right["bybit_funding_source_timestamp"] = pd.to_datetime(
            right["bybit_funding_source_timestamp"], utc=True, errors="coerce"
        )
        return pd.merge_asof(
            left.dropna(subset=["timestamp"]).sort_values("timestamp"),
            right.dropna(subset=["bybit_funding_source_timestamp"]).sort_values("bybit_funding_source_timestamp"),
            left_on="timestamp",
            right_on="bybit_funding_source_timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
