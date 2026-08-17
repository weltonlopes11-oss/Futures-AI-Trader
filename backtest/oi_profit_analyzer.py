from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

import pandas as pd


@dataclass(slots=True)
class OIProfitObservation:
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    decision: str
    regime: str
    market_direction: str
    oi_change_pct: float
    oi_bucket: str
    entry_price: float
    horizon_minutes: int
    future_return_pct: float
    directional_return_pct: float
    net_return_pct: float
    mfe_pct: float
    mae_pct: float


class OIProfitAnalyzer:
    """Analisa o potencial econômico dos impulsos de Open Interest.

    A entrada é simulada na abertura do candle seguinte ao sinal, evitando usar
    o fechamento do próprio candle que gerou a decisão. Custos de transação são
    configuráveis e aplicados apenas a LONG/SHORT.
    """

    DEFAULT_HORIZONS = (5, 15, 30, 60)

    def __init__(
        self,
        records: Iterable,
        candles: pd.DataFrame,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        fee_bps_per_side: float = 0.0,
        slippage_bps_per_side: float = 0.0,
    ):
        self.records = list(records)
        self.candles = self._prepare_candles(candles)
        self.horizons = tuple(sorted(set(int(x) for x in horizons if int(x) > 0)))
        self.fee_bps_per_side = max(float(fee_bps_per_side), 0.0)
        self.slippage_bps_per_side = max(float(slippage_bps_per_side), 0.0)

        if not self.horizons:
            raise ValueError("Informe pelo menos um horizonte positivo.")

        self.round_trip_cost_pct = (
            2.0 * (self.fee_bps_per_side + self.slippage_bps_per_side) / 100.0
        )

        self.base_observations = self._build_base_observations()
        self.thresholds = self._build_thresholds()
        self.observations = self._build_profit_observations()

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if isfinite(number) else None

    @staticmethod
    def _prepare_candles(candles: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(candles, pd.DataFrame):
            raise TypeError("candles deve ser um DataFrame.")

        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(candles.columns)
        if missing:
            raise ValueError(f"Candles sem colunas obrigatórias: {sorted(missing)}")

        frame = candles.copy().reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).astype("datetime64[ns]")

        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")

        return (
            frame.drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _build_base_observations(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        previous_oi: float | None = None

        for record in self.records:
            metadata = record.market.metadata or {}
            oi = self._number(metadata.get("open_interest"))
            if oi is None:
                continue

            if previous_oi is not None and oi == previous_oi:
                continue

            change = self._number(metadata.get("open_interest_change_pct"))
            if change is None and previous_oi not in (None, 0):
                change = ((oi / previous_oi) - 1.0) * 100.0

            previous_oi = oi
            if change is None:
                continue

            result.append(
                {
                    "timestamp": pd.Timestamp(record.timestamp),
                    "decision": record.decision.action,
                    "regime": record.market.regime,
                    "market_direction": record.market.direction,
                    "oi_change_pct": change,
                }
            )

        return result

    def _build_thresholds(self) -> dict[str, float]:
        changes = sorted(item["oi_change_pct"] for item in self.base_observations)
        if not changes:
            return {"p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}

        return {
            "p10": self._percentile(changes, 0.10),
            "p25": self._percentile(changes, 0.25),
            "p75": self._percentile(changes, 0.75),
            "p90": self._percentile(changes, 0.90),
        }

    def bucket_for(self, change: float) -> str:
        if change <= self.thresholds["p10"]:
            return "STRONG_DROP"
        if change <= self.thresholds["p25"]:
            return "DROP"
        if change < self.thresholds["p75"]:
            return "NORMAL"
        if change < self.thresholds["p90"]:
            return "RISE"
        return "STRONG_RISE"

    def _build_profit_observations(self) -> list[OIProfitObservation]:
        result: list[OIProfitObservation] = []

        for item in self.base_observations:
            signal_timestamp = item["timestamp"]
            future_all = self.candles.loc[
                self.candles["timestamp"] > signal_timestamp
            ]

            if future_all.empty:
                continue

            for horizon in self.horizons:
                future = future_all.head(horizon)
                if len(future) < horizon:
                    continue

                entry_timestamp = pd.Timestamp(future.iloc[0]["timestamp"])
                entry_price = self._number(future.iloc[0]["open"])
                exit_price = self._number(future.iloc[-1]["close"])
                high_price = self._number(future["high"].max())
                low_price = self._number(future["low"].min())

                if None in (entry_price, exit_price, high_price, low_price) or entry_price == 0:
                    continue

                future_return = ((exit_price / entry_price) - 1.0) * 100.0
                decision = str(item["decision"])

                if decision == "LONG":
                    directional_return = future_return
                    mfe = ((high_price / entry_price) - 1.0) * 100.0
                    mae = ((entry_price / low_price) - 1.0) * 100.0
                    net_return = directional_return - self.round_trip_cost_pct
                elif decision == "SHORT":
                    directional_return = -future_return
                    mfe = ((entry_price / low_price) - 1.0) * 100.0
                    mae = ((high_price / entry_price) - 1.0) * 100.0
                    net_return = directional_return - self.round_trip_cost_pct
                else:
                    directional_return = 0.0
                    net_return = 0.0
                    mfe = max(
                        ((high_price / entry_price) - 1.0) * 100.0,
                        ((entry_price / low_price) - 1.0) * 100.0,
                    )
                    mae = min(
                        ((high_price / entry_price) - 1.0) * 100.0,
                        ((entry_price / low_price) - 1.0) * 100.0,
                    )

                result.append(
                    OIProfitObservation(
                        signal_timestamp=signal_timestamp,
                        entry_timestamp=entry_timestamp,
                        decision=decision,
                        regime=str(item["regime"]),
                        market_direction=str(item["market_direction"]),
                        oi_change_pct=float(item["oi_change_pct"]),
                        oi_bucket=self.bucket_for(float(item["oi_change_pct"])),
                        entry_price=float(entry_price),
                        horizon_minutes=horizon,
                        future_return_pct=float(future_return),
                        directional_return_pct=float(directional_return),
                        net_return_pct=float(net_return),
                        mfe_pct=max(float(mfe), 0.0),
                        mae_pct=max(float(mae), 0.0),
                    )
                )

        return result

    def summary_by_bucket(self):
        grouped = defaultdict(lambda: defaultdict(list))
        for item in self.observations:
            grouped[item.horizon_minutes][item.oi_bucket].append(item)
        return {
            horizon: {bucket: self._summarize(items) for bucket, items in buckets.items()}
            for horizon, buckets in grouped.items()
        }

    def summary_by_decision_and_bucket(self):
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for item in self.observations:
            grouped[item.horizon_minutes][item.decision][item.oi_bucket].append(item)
        return {
            horizon: {
                decision: {bucket: self._summarize(items) for bucket, items in buckets.items()}
                for decision, buckets in decisions.items()
            }
            for horizon, decisions in grouped.items()
        }

    def summary_by_regime_and_bucket(self):
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for item in self.observations:
            grouped[item.horizon_minutes][item.regime][item.oi_bucket].append(item)
        return {
            horizon: {
                regime: {bucket: self._summarize(items) for bucket, items in buckets.items()}
                for regime, buckets in regimes.items()
            }
            for horizon, regimes in grouped.items()
        }

    @staticmethod
    def _summarize(items: list[OIProfitObservation]) -> dict[str, float | int]:
        if not items:
            return {
                "count": 0,
                "avg_directional_return_pct": 0.0,
                "avg_net_return_pct": 0.0,
                "median_net_return_pct": 0.0,
                "net_positive_rate_pct": 0.0,
                "avg_mfe_pct": 0.0,
                "avg_mae_pct": 0.0,
                "mfe_mae_ratio": 0.0,
            }

        directional = [item.directional_return_pct for item in items]
        net = sorted(item.net_return_pct for item in items)
        mfes = [item.mfe_pct for item in items]
        maes = [item.mae_pct for item in items]

        n = len(items)
        avg_directional = sum(directional) / n
        avg_net = sum(net) / n
        avg_mfe = sum(mfes) / n
        avg_mae = sum(maes) / n
        positive_rate = sum(1 for value in net if value > 0) / n * 100.0

        median_net = (
            net[n // 2]
            if n % 2
            else (net[n // 2 - 1] + net[n // 2]) / 2.0
        )

        return {
            "count": n,
            "avg_directional_return_pct": round(avg_directional, 6),
            "avg_net_return_pct": round(avg_net, 6),
            "median_net_return_pct": round(median_net, 6),
            "net_positive_rate_pct": round(positive_rate, 2),
            "avg_mfe_pct": round(avg_mfe, 6),
            "avg_mae_pct": round(avg_mae, 6),
            "mfe_mae_ratio": round(avg_mfe / avg_mae if avg_mae > 0 else 0.0, 4),
        }

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight
