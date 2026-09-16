from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiquiditySignalConfig:
    sample_seconds: int = 2
    windows_seconds: tuple[int, ...] = (10, 30, 60)
    max_spread_bps: float = 5.0
    min_samples_ratio: float = 0.8
    score_threshold: float = 0.25


class LiquiditySignalEngine:
    """Turn raw order-book snapshots into a stable causal liquidity score.

    The score combines five independent dimensions without looking ahead:
      35% persistent depth imbalance
      25% persistent microprice edge direction
      20% depth-imbalance acceleration
      10% spread/execution quality
      10% directional stability

    The output is intentionally suitable for shadow-mode validation first.
    No trade is created by this component; it only confirms, blocks or stays
    neutral relative to an upstream strategy decision.
    """

    def __init__(self, config: LiquiditySignalConfig | None = None):
        self.config = config or LiquiditySignalConfig()
        if self.config.sample_seconds <= 0:
            raise ValueError("sample_seconds must be positive")
        if not self.config.windows_seconds:
            raise ValueError("windows_seconds cannot be empty")
        if any(w <= 0 for w in self.config.windows_seconds):
            raise ValueError("all windows must be positive")
        if self.config.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive")
        if not 0 < self.config.min_samples_ratio <= 1:
            raise ValueError("min_samples_ratio must be in (0, 1]")
        if not 0 <= self.config.score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")

    @staticmethod
    def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return float(np.clip(value, lo, hi))

    def _window_slice(self, frame: pd.DataFrame, end_ts: pd.Timestamp, seconds: int) -> pd.DataFrame:
        start_ts = end_ts - pd.Timedelta(seconds=seconds)
        return frame[(frame["timestamp"] > start_ts) & (frame["timestamp"] <= end_ts)]

    def _min_samples(self, seconds: int) -> int:
        expected = max(1, int(seconds / self.config.sample_seconds))
        return max(1, int(np.ceil(expected * self.config.min_samples_ratio)))

    def score_at(self, snapshots: pd.DataFrame, at: pd.Timestamp | None = None) -> dict:
        required = {
            "timestamp",
            "spread_bps",
            "depth_imbalance",
            "microprice_edge_bps",
        }
        missing = required.difference(snapshots.columns)
        if missing:
            raise ValueError(f"missing liquidity columns: {sorted(missing)}")

        frame = snapshots.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            raise ValueError("snapshots cannot be empty")

        if at is None:
            at = frame["timestamp"].iloc[-1]
        else:
            at = pd.Timestamp(at)
            if at.tzinfo is None:
                at = at.tz_localize("UTC")
            else:
                at = at.tz_convert("UTC")

        frame = frame[frame["timestamp"] <= at]
        if frame.empty:
            raise ValueError("no snapshots available at requested timestamp")

        windows = sorted(self.config.windows_seconds)
        stats: dict[str, float | int] = {}
        window_means: list[float] = []
        micro_signs: list[float] = []
        stability_values: list[float] = []

        for seconds in windows:
            sample = self._window_slice(frame, at, seconds)
            count = len(sample)
            stats[f"samples_{seconds}s"] = count
            if count < self._min_samples(seconds):
                raise ValueError(f"insufficient samples for {seconds}s window: {count}")

            imbalance_mean = float(sample["depth_imbalance"].mean())
            micro_mean = float(sample["microprice_edge_bps"].mean())
            positive_share = float((sample["depth_imbalance"] > 0).mean())
            negative_share = float((sample["depth_imbalance"] < 0).mean())
            directional_share = max(positive_share, negative_share)
            signed_stability = directional_share * (1.0 if imbalance_mean >= 0 else -1.0)

            stats[f"imbalance_mean_{seconds}s"] = imbalance_mean
            stats[f"microprice_mean_{seconds}s"] = micro_mean
            stats[f"positive_share_{seconds}s"] = positive_share
            stats[f"negative_share_{seconds}s"] = negative_share
            window_means.append(imbalance_mean)
            micro_signs.append(np.sign(micro_mean))
            stability_values.append(signed_stability)

        persistent_imbalance = self._clip(float(np.mean(window_means)))

        micro_scale = max(
            0.01,
            float(frame["microprice_edge_bps"].abs().quantile(0.90)),
        )
        micro_component = self._clip(
            float(np.mean([stats[f"microprice_mean_{w}s"] for w in windows])) / micro_scale
        )

        shortest = window_means[0]
        longest = window_means[-1]
        acceleration = self._clip(shortest - longest)

        latest_spread = float(frame.iloc[-1]["spread_bps"])
        if latest_spread >= self.config.max_spread_bps:
            spread_quality = -1.0
        else:
            spread_quality = self._clip(1.0 - latest_spread / self.config.max_spread_bps, 0.0, 1.0)
            direction = np.sign(persistent_imbalance) or 1.0
            spread_quality *= float(direction)

        stability = self._clip(float(np.mean(stability_values)))

        score = self._clip(
            0.35 * persistent_imbalance
            + 0.25 * micro_component
            + 0.20 * acceleration
            + 0.10 * spread_quality
            + 0.10 * stability
        )

        blocked = latest_spread > self.config.max_spread_bps
        if blocked:
            signal = "BLOCK"
        elif score >= self.config.score_threshold:
            signal = "LONG"
        elif score <= -self.config.score_threshold:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"

        return {
            "timestamp": at,
            "liquidity_score": score,
            "liquidity_signal": signal,
            "persistent_imbalance": persistent_imbalance,
            "microprice_component": micro_component,
            "imbalance_acceleration": acceleration,
            "spread_quality": spread_quality,
            "directional_stability": stability,
            "spread_bps": latest_spread,
            **stats,
        }

    def enrich(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        frame = snapshots.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        rows: list[dict] = []
        for ts in frame["timestamp"]:
            try:
                rows.append(self.score_at(frame, at=ts))
            except ValueError:
                continue
        return pd.DataFrame(rows)
