from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtest.order_book_liquidity import OrderBookLiquidity, OrderBookLiquidityConfig


def _snapshot(bids, asks):
    return {
        "lastUpdateId": 123,
        "E": 1789578050623,
        "T": 1789578050620,
        "bids": [[str(p), str(q)] for p, q in bids],
        "asks": [[str(p), str(q)] for p, q in asks],
    }


def test_long_signal_when_bid_depth_and_microprice_agree():
    snap = _snapshot(
        bids=[(100.0, 10.0), (99.9, 8.0), (99.8, 6.0)],
        asks=[(100.1, 2.0), (100.2, 2.0), (100.3, 2.0)],
    )
    engine = OrderBookLiquidity(OrderBookLiquidityConfig(depth_levels=3, max_spread_bps=20))
    row = engine.from_snapshot(snap)

    assert row["depth_imbalance"] > 0
    assert row["microprice_edge_bps"] > 0
    assert row["liquidity_signal"] == "LONG"


def test_short_signal_when_ask_depth_and_microprice_agree():
    snap = _snapshot(
        bids=[(100.0, 2.0), (99.9, 2.0), (99.8, 2.0)],
        asks=[(100.1, 10.0), (100.2, 8.0), (100.3, 6.0)],
    )
    engine = OrderBookLiquidity(OrderBookLiquidityConfig(depth_levels=3, max_spread_bps=20))
    row = engine.from_snapshot(snap)

    assert row["depth_imbalance"] < 0
    assert row["microprice_edge_bps"] < 0
    assert row["liquidity_signal"] == "SHORT"


def test_neutral_when_depth_and_microprice_disagree():
    snap = _snapshot(
        bids=[(100.0, 1.0), (99.9, 20.0)],
        asks=[(100.1, 5.0), (100.2, 1.0)],
    )
    engine = OrderBookLiquidity(OrderBookLiquidityConfig(depth_levels=2, max_spread_bps=20))
    row = engine.from_snapshot(snap)

    assert row["depth_imbalance"] > 0
    assert row["microprice_edge_bps"] < 0
    assert row["liquidity_signal"] == "NEUTRAL"


def test_blocks_when_spread_is_too_wide():
    snap = _snapshot(
        bids=[(100.0, 10.0)],
        asks=[(101.0, 1.0)],
    )
    engine = OrderBookLiquidity(OrderBookLiquidityConfig(depth_levels=1, max_spread_bps=5))
    row = engine.from_snapshot(snap)
    assert row["spread_bps"] > 5
    assert row["liquidity_signal"] == "BLOCK"


def test_timestamp_can_be_supplied_explicitly():
    snap = _snapshot(bids=[(100.0, 1.0)], asks=[(100.1, 1.0)])
    ts = datetime(2026, 9, 16, 16, 0, tzinfo=timezone.utc)
    row = OrderBookLiquidity().from_snapshot(snap, observed_at=ts)
    assert row["timestamp"].to_pydatetime() == ts


def test_rejects_crossed_book():
    snap = _snapshot(bids=[(100.2, 1.0)], asks=[(100.1, 1.0)])
    with pytest.raises(ValueError, match="crossed"):
        OrderBookLiquidity().from_snapshot(snap)
