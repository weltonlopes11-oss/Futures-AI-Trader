from __future__ import annotations

import csv
import os
import signal
import time
from pathlib import Path

import pandas as pd
import requests

from backtest.liquidity_signal_engine import LiquiditySignalConfig, LiquiditySignalEngine
from backtest.order_book_liquidity import OrderBookLiquidity, OrderBookLiquidityConfig


BINANCE_FAPI_DEPTH = "https://fapi.binance.com/fapi/v1/depth"


class StopRequested(Exception):
    pass


def fetch_snapshot(symbol: str, limit: int, session: requests.Session) -> dict:
    response = session.get(
        BINANCE_FAPI_DEPTH,
        params={"symbol": symbol, "limit": limit},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_recent(path: Path, seconds: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        return frame
    cutoff = frame["timestamp"].max() - pd.Timedelta(seconds=seconds + 10)
    return frame[frame["timestamp"] >= cutoff].reset_index(drop=True)


def main() -> None:
    symbol = os.getenv("ORDER_BOOK_SYMBOL", "ETHUSDT").upper()
    api_limit = int(os.getenv("ORDER_BOOK_API_LIMIT", "100"))
    depth_levels = int(os.getenv("ORDER_BOOK_DEPTH_LEVELS", "50"))
    interval_seconds = float(os.getenv("ORDER_BOOK_SAMPLE_SECONDS", "2"))
    max_spread_bps = float(os.getenv("ORDER_BOOK_MAX_SPREAD_BPS", "5"))
    score_threshold = float(os.getenv("LIQUIDITY_SCORE_THRESHOLD", "0.25"))

    raw_path = Path(
        os.getenv(
            "ORDER_BOOK_OUTPUT",
            f"research_snapshots/order_book/{symbol}_raw.csv",
        )
    )
    score_path = Path(
        os.getenv(
            "LIQUIDITY_SCORE_OUTPUT",
            f"research_snapshots/order_book/{symbol}_scores.csv",
        )
    )

    snapshot_engine = OrderBookLiquidity(
        OrderBookLiquidityConfig(
            depth_levels=depth_levels,
            max_spread_bps=max_spread_bps,
        )
    )
    score_engine = LiquiditySignalEngine(
        LiquiditySignalConfig(
            sample_seconds=max(1, int(round(interval_seconds))),
            windows_seconds=(10, 30, 60),
            max_spread_bps=max_spread_bps,
            score_threshold=score_threshold,
        )
    )

    session = requests.Session()
    stopping = False

    def request_stop(signum, frame):  # noqa: ARG001
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    failures = 0
    print(
        f"Starting {symbol} order-book collector every {interval_seconds:.1f}s "
        f"using {depth_levels} levels. raw={raw_path} scores={score_path}"
    )

    while not stopping:
        started = time.monotonic()
        try:
            snapshot = fetch_snapshot(symbol, api_limit, session)
            row = snapshot_engine.from_snapshot(snapshot)
            row["timestamp"] = row["timestamp"].isoformat()
            row["symbol"] = symbol
            append_row(raw_path, row)
            failures = 0

            recent = read_recent(raw_path, 60)
            if not recent.empty:
                try:
                    score = score_engine.score_at(recent)
                except ValueError:
                    score = None
                if score is not None:
                    score_row = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in score.items()}
                    score_row["symbol"] = symbol
                    append_row(score_path, score_row)
                    print(
                        f"{score_row['timestamp']} {symbol} "
                        f"score={score_row['liquidity_score']:+.3f} "
                        f"signal={score_row['liquidity_signal']} "
                        f"imbalance={score_row['persistent_imbalance']:+.3f} "
                        f"accel={score_row['imbalance_acceleration']:+.3f} "
                        f"spread={score_row['spread_bps']:.4f}bps"
                    )
        except requests.RequestException as exc:
            failures += 1
            backoff = min(30.0, max(interval_seconds, 2.0 ** min(failures, 5)))
            print(f"order-book request failed ({failures}): {exc}; backoff={backoff:.1f}s")
            time.sleep(backoff)
            continue
        except Exception as exc:
            failures += 1
            print(f"collector error ({failures}): {exc}")

        elapsed = time.monotonic() - started
        sleep_for = max(0.0, interval_seconds - elapsed)
        if sleep_for:
            time.sleep(sleep_for)

    print("Order-book collector stopped cleanly.")


if __name__ == "__main__":
    main()
