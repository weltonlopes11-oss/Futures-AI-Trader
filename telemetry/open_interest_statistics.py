from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite
from statistics import mean, median
from typing import Any


class OpenInterestStatistics:
    """Estatísticas exploratórias do Open Interest sem alterar decisões.

    A análise trabalha por observações reais de OI, e não por candle de 1 minuto.
    Como o OI de 5m é propagado causalmente até a próxima observação, contar todos
    os candles duplicaria artificialmente cada leitura.
    """

    def __init__(self, records):
        self.records = records
        self.observations = self._build_observations()

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if isfinite(number) else None

    @staticmethod
    def _direction(value: float) -> str:
        if value > 0:
            return "UP"
        if value < 0:
            return "DOWN"
        return "STABLE"

    def _build_observations(self) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        previous_oi: float | None = None
        previous_price: float | None = None

        for record in self.records:
            metadata = record.market.metadata or {}
            oi = self._number(metadata.get("open_interest"))

            if oi is None:
                continue

            # O merge causal mantém a última observação de OI nos candles seguintes.
            # Só consideramos um novo ponto quando o valor efetivamente muda.
            if previous_oi is not None and oi == previous_oi:
                continue

            price = self._number(record.close_price)
            api_change = self._number(metadata.get("open_interest_change_pct"))

            if previous_oi is None:
                oi_change_pct = api_change
                price_change_pct = None
            else:
                oi_change_pct = (
                    ((oi / previous_oi) - 1.0) * 100.0
                    if previous_oi != 0
                    else None
                )
                price_change_pct = (
                    ((price / previous_price) - 1.0) * 100.0
                    if price is not None
                    and previous_price not in (None, 0)
                    else None
                )

            observations.append(
                {
                    "timestamp": record.timestamp,
                    "decision": record.decision.action,
                    "regime": record.market.regime,
                    "market_direction": record.market.direction,
                    "open_interest": oi,
                    "oi_change_pct": oi_change_pct,
                    "price": price,
                    "price_change_pct": price_change_pct,
                }
            )

            previous_oi = oi
            if price is not None:
                previous_price = price

        return observations

    @property
    def total_observations(self) -> int:
        return len(self.observations)

    def oi_direction_distribution(self) -> dict[str, int]:
        counter = Counter()
        for item in self.observations:
            change = item["oi_change_pct"]
            if change is not None:
                counter[self._direction(change)] += 1
        return dict(counter)

    def by_decision(self) -> dict[str, dict[str, int]]:
        result: dict[str, Counter] = defaultdict(Counter)
        for item in self.observations:
            change = item["oi_change_pct"]
            if change is None:
                continue
            result[item["decision"]][self._direction(change)] += 1
        return {key: dict(value) for key, value in result.items()}

    def by_regime(self) -> dict[str, dict[str, int]]:
        result: dict[str, Counter] = defaultdict(Counter)
        for item in self.observations:
            change = item["oi_change_pct"]
            if change is None:
                continue
            result[item["regime"]][self._direction(change)] += 1
        return {key: dict(value) for key, value in result.items()}

    def price_oi_combinations(self) -> dict[str, int]:
        counter = Counter()
        for item in self.observations:
            oi_change = item["oi_change_pct"]
            price_change = item["price_change_pct"]
            if oi_change is None or price_change is None:
                continue
            key = (
                f"PRICE_{self._direction(price_change)}"
                f"__OI_{self._direction(oi_change)}"
            )
            counter[key] += 1
        return dict(counter)

    def change_statistics(self) -> dict[str, dict[str, float | int]]:
        return {
            "oi": self._describe(
                item["oi_change_pct"]
                for item in self.observations
            ),
            "price": self._describe(
                item["price_change_pct"]
                for item in self.observations
            ),
        }

    def decision_change_statistics(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for item in self.observations:
            change = item["oi_change_pct"]
            if change is not None:
                grouped[item["decision"]].append(change)
        return {
            decision: self._describe(values)
            for decision, values in grouped.items()
        }

    @classmethod
    def _describe(cls, values) -> dict[str, float | int]:
        clean = sorted(
            value
            for raw in values
            if (value := cls._number(raw)) is not None
        )

        if not clean:
            return {
                "count": 0,
                "average": 0.0,
                "median": 0.0,
                "min": 0.0,
                "p10": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "max": 0.0,
            }

        return {
            "count": len(clean),
            "average": round(mean(clean), 6),
            "median": round(median(clean), 6),
            "min": round(clean[0], 6),
            "p10": round(cls._percentile(clean, 0.10), 6),
            "p25": round(cls._percentile(clean, 0.25), 6),
            "p75": round(cls._percentile(clean, 0.75), 6),
            "p90": round(cls._percentile(clean, 0.90), 6),
            "max": round(clean[-1], 6),
        }

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight
