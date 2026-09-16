from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OrderBookLiquidityConfig:
    depth_levels: int = 50
    max_spread_bps: float = 5.0


class OrderBookLiquidity:
    """Causal order-book liquidity features from one Binance USD-M snapshot.

    The component deliberately avoids tuned directional thresholds. It uses:
      - quoted bid/ask notional across the first N levels;
      - depth imbalance = (bid - ask) / (bid + ask);
      - top-of-book microprice edge versus mid;
      - spread as an execution-quality guard.

    Direction is confirmed only when depth imbalance and microprice agree.
    Historical reconstruction is intentionally not attempted because Binance's
    public futures depth endpoint exposes the current book, not an archive of
    past snapshots.
    """

    def __init__(self, config: OrderBookLiquidityConfig | None = None):
        self.config = config or OrderBookLiquidityConfig()
        if self.config.depth_levels <= 0:
            raise ValueError("depth_levels must be positive")
        if self.config.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive")

    @staticmethod
    def _levels(raw: list[list[str]]) -> list[tuple[float, float]]:
        levels: list[tuple[float, float]] = []
        for row in raw:
            if len(row) < 2:
                continue
            price = float(row[0])
            qty = float(row[1])
            if price <= 0 or qty < 0:
                raise ValueError("invalid order-book level")
            levels.append((price, qty))
        return levels

    def from_snapshot(
        self,
        snapshot: dict[str, Any],
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        bids = self._levels(snapshot.get("bids", []))
        asks = self._levels(snapshot.get("asks", []))
        if not bids or not asks:
            raise ValueError("snapshot must contain bids and asks")

        bids = bids[: self.config.depth_levels]
        asks = asks[: self.config.depth_levels]

        best_bid, best_bid_qty = bids[0]
        best_ask, best_ask_qty = asks[0]
        if best_bid >= best_ask:
            raise ValueError("crossed or invalid order book")

        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 10_000.0

        bid_depth_quote = sum(price * qty for price, qty in bids)
        ask_depth_quote = sum(price * qty for price, qty in asks)
        total_depth_quote = bid_depth_quote + ask_depth_quote
        depth_imbalance = (
            (bid_depth_quote - ask_depth_quote) / total_depth_quote
            if total_depth_quote > 0
            else 0.0
        )

        top_qty = best_bid_qty + best_ask_qty
        top_imbalance = (
            (best_bid_qty - best_ask_qty) / top_qty if top_qty > 0 else 0.0
        )

        # Standard microprice weights each side by the opposite queue size.
        microprice = (
            (best_ask * best_bid_qty + best_bid * best_ask_qty) / top_qty
            if top_qty > 0
            else mid
        )
        microprice_edge_bps = (microprice - mid) / mid * 10_000.0

        if observed_at is None:
            event_ms = snapshot.get("E") or snapshot.get("T")
            if event_ms is not None:
                observed_at = datetime.fromtimestamp(float(event_ms) / 1000.0, tz=timezone.utc)
            else:
                observed_at = datetime.now(timezone.utc)
        elif observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)

        if spread_bps > self.config.max_spread_bps:
            liquidity_signal = "BLOCK"
        elif depth_imbalance > 0 and microprice_edge_bps > 0:
            liquidity_signal = "LONG"
        elif depth_imbalance < 0 and microprice_edge_bps < 0:
            liquidity_signal = "SHORT"
        else:
            liquidity_signal = "NEUTRAL"

        return {
            "timestamp": pd.Timestamp(observed_at),
            "last_update_id": snapshot.get("lastUpdateId"),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": spread_bps,
            "bid_depth_quote": bid_depth_quote,
            "ask_depth_quote": ask_depth_quote,
            "depth_imbalance": depth_imbalance,
            "top_imbalance": top_imbalance,
            "microprice": microprice,
            "microprice_edge_bps": microprice_edge_bps,
            "liquidity_signal": liquidity_signal,
            "depth_levels": min(len(bids), len(asks)),
        }

    def to_frame(self, snapshots: list[dict[str, Any]]) -> pd.DataFrame:
        rows = [self.from_snapshot(snapshot) for snapshot in snapshots]
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
