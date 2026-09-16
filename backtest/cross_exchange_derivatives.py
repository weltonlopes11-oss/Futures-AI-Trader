from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class CrossExchangeConfig:
    timeout_seconds: int = 15


class CrossExchangeDerivatives:
    """Public, read-only cross-exchange derivatives snapshot.

    Native OI units differ between venues. The collector only computes an
    aggregate USD OI for venues where a defensible base-asset OI conversion is
    available in the public response. Raw/native values are always preserved.
    """

    def __init__(self, config: CrossExchangeConfig | None = None):
        self.config = config or CrossExchangeConfig()

    def _get(self, url: str, params: dict[str, Any]) -> Any:
        r = requests.get(url, params=params, timeout=self.config.timeout_seconds)
        r.raise_for_status()
        return r.json()

    def binance(self, symbol: str = "ETHUSDT") -> dict[str, Any]:
        oi = self._get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            {"symbol": symbol},
        )
        premium = self._get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            {"symbol": symbol},
        )
        mark = float(premium["markPrice"])
        oi_base = float(oi["openInterest"])
        return {
            "exchange": "binance",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc),
            "oi_native": oi_base,
            "oi_base": oi_base,
            "oi_usd_estimate": oi_base * mark,
            "funding_rate": float(premium.get("lastFundingRate", 0.0)),
            "mark_price": mark,
        }

    def bybit(self, symbol: str = "ETHUSDT") -> dict[str, Any]:
        oi_payload = self._get(
            "https://api.bybit.com/v5/market/open-interest",
            {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": "5min",
                "limit": 1,
            },
        )
        ticker_payload = self._get(
            "https://api.bybit.com/v5/market/tickers",
            {"category": "linear", "symbol": symbol},
        )
        oi_row = oi_payload["result"]["list"][0]
        ticker = ticker_payload["result"]["list"][0]
        oi_base = float(oi_row["openInterest"])
        mark = float(ticker["markPrice"])
        return {
            "exchange": "bybit",
            "symbol": symbol,
            "timestamp": pd.to_datetime(int(oi_row["timestamp"]), unit="ms", utc=True).to_pydatetime(),
            "oi_native": oi_base,
            "oi_base": oi_base,
            "oi_usd_estimate": oi_base * mark,
            "funding_rate": float(ticker.get("fundingRate", 0.0)),
            "mark_price": mark,
        }

    def okx(self, instrument: str = "ETH-USDT-SWAP") -> dict[str, Any]:
        oi_payload = self._get(
            "https://www.okx.com/api/v5/public/open-interest",
            {"instType": "SWAP", "instId": instrument},
        )
        funding_payload = self._get(
            "https://www.okx.com/api/v5/public/funding-rate",
            {"instId": instrument},
        )
        ticker_payload = self._get(
            "https://www.okx.com/api/v5/market/ticker",
            {"instId": instrument},
        )
        oi_row = oi_payload["data"][0]
        funding = funding_payload["data"][0]
        ticker = ticker_payload["data"][0]
        mark = float(ticker["last"])
        oi_native = float(oi_row["oi"])
        oi_base = float(oi_row.get("oiCcy") or 0.0)
        return {
            "exchange": "okx",
            "symbol": instrument,
            "timestamp": pd.to_datetime(int(oi_row["ts"]), unit="ms", utc=True).to_pydatetime(),
            "oi_native": oi_native,
            "oi_base": oi_base if oi_base > 0 else None,
            "oi_usd_estimate": oi_base * mark if oi_base > 0 else None,
            "funding_rate": float(funding.get("fundingRate", 0.0)),
            "mark_price": mark,
        }

    def deribit(self, instrument: str = "ETH-PERPETUAL") -> dict[str, Any]:
        payload = self._get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_instrument",
            {"instrument_name": instrument},
        )
        row = payload["result"][0]
        return {
            "exchange": "deribit",
            "symbol": instrument,
            "timestamp": pd.to_datetime(int(row["creation_timestamp"]), unit="ms", utc=True).to_pydatetime(),
            "oi_native": float(row.get("open_interest", 0.0)),
            "oi_base": None,
            "oi_usd_estimate": None,
            "funding_rate": float(row.get("funding_8h") or row.get("current_funding") or 0.0),
            "mark_price": float(row.get("mark_price") or 0.0),
        }

    def snapshot(self) -> pd.DataFrame:
        rows = []
        errors = []
        for name, fn in (
            ("binance", self.binance),
            ("bybit", self.bybit),
            ("okx", self.okx),
            ("deribit", self.deribit),
        ):
            try:
                rows.append(fn())
            except Exception as exc:  # one venue must not invalidate all others
                errors.append({"exchange": name, "error": str(exc)})
        frame = pd.DataFrame(rows)
        frame.attrs["errors"] = errors
        return frame

    @staticmethod
    def aggregate(frame: pd.DataFrame) -> dict[str, float]:
        if frame.empty:
            return {"global_oi_usd": float("nan"), "oi_weighted_funding": float("nan")}
        valid = frame.dropna(subset=["oi_usd_estimate", "funding_rate"]).copy()
        if valid.empty or valid["oi_usd_estimate"].sum() <= 0:
            return {"global_oi_usd": float("nan"), "oi_weighted_funding": float("nan")}
        total = float(valid["oi_usd_estimate"].sum())
        weighted = float((valid["funding_rate"] * valid["oi_usd_estimate"]).sum() / total)
        return {"global_oi_usd": total, "oi_weighted_funding": weighted}
