from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

SYMBOL = "ETHUSDT"
INTERVAL = "1h"
EXPECTED_SOURCE = "binance-usdm-futures"
DIRECT_ENDPOINTS = (
    "https://fapi.binance.com/fapi/v1/klines",
    "https://fapi1.binance.com/fapi/v1/klines",
    "https://fapi2.binance.com/fapi/v1/klines",
)


def _validated_json_array(payload: bytes, source: str) -> list:
    if not payload.strip():
        raise RuntimeError(f"empty response from {source}")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response from {source}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError(f"unexpected response from {source}")
    if any(not isinstance(row, list) or len(row) < 12 for row in parsed):
        raise RuntimeError(f"invalid kline schema from {source}")
    return parsed


def _fetch_collector(limit: int) -> list:
    base_url = os.environ.get("MARKET_DATA_COLLECTOR_URL", "").strip()
    api_key = os.environ.get("MARKET_DATA_COLLECTOR_KEY", "").strip()
    if not base_url:
        raise RuntimeError("MARKET_DATA_COLLECTOR_URL is not configured")
    if not api_key:
        raise RuntimeError("MARKET_DATA_COLLECTOR_KEY is not configured")

    query = urllib.parse.urlencode({"symbol": SYMBOL, "interval": INTERVAL, "limit": limit})
    separator = "&" if "?" in base_url else "?"
    req = urllib.request.Request(
        f"{base_url}{separator}{query}",
        headers={
            "User-Agent": "Futures-AI-Trader/1.0",
            "Accept": "application/json",
            "X-Collector-Key": api_key,
        },
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        source = response.headers.get("X-Market-Source", "")
        if source != EXPECTED_SOURCE:
            raise RuntimeError(f"collector source attestation failed: {source or 'missing'}")
        payload = response.read()
    return _validated_json_array(payload, "market-data-collector")


def _fetch_direct(limit: int) -> list:
    query = urllib.parse.urlencode({"symbol": SYMBOL, "interval": INTERVAL, "limit": limit})
    errors: list[str] = []
    for endpoint in DIRECT_ENDPOINTS:
        try:
            req = urllib.request.Request(
                f"{endpoint}?{query}",
                headers={"User-Agent": "Futures-AI-Trader/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                payload = response.read()
            return _validated_json_array(payload, urllib.parse.urlparse(endpoint).netloc)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            code = getattr(exc, "code", "network")
            errors.append(f"{urllib.parse.urlparse(endpoint).netloc}:{code}:{type(exc).__name__}")
    raise RuntimeError("All Binance USD-M Futures endpoints failed: " + ", ".join(errors))


def fetch_klines(limit: int = 200) -> pd.DataFrame:
    # Production/forward mode uses the authenticated collector whenever configured.
    # Direct Binance access remains only for local development and diagnostics.
    if os.environ.get("MARKET_DATA_COLLECTOR_URL", "").strip():
        raw = _fetch_collector(limit)
    else:
        raw = _fetch_direct(limit)

    rows = []
    for k in raw:
        rows.append({
            "open_time_ms": int(k[0]),
            "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True).tz_localize(None),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "close_time_ms": int(k[6]),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("market-data adapter returned no candles")
    if frame["open_time_ms"].duplicated().any():
        raise RuntimeError("duplicate candle open times")
    if not frame["open_time_ms"].is_monotonic_increasing:
        raise RuntimeError("candles are not ordered by open time")
    return frame
