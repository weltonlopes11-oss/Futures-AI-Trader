from __future__ import annotations

from collections import Counter

from telemetry.telemetry_models import TelemetryRecord


class TelemetryStatistics:
    """
    Consolida estatísticas dos registros
    coletados durante o backtest.

    Esta classe NÃO imprime relatórios.

    Apenas produz dados agregados.
    """

    def __init__(
        self,
        records: list[TelemetryRecord],
    ):

        self.records = records

    # ==========================================================
    # Quantidade
    # ==========================================================

    @property
    def total_records(self) -> int:

        return len(self.records)

    # ==========================================================
    # Decisões
    # ==========================================================

    def decision_distribution(self) -> dict[str, int]:

        return dict(

            Counter(

                record.decision.action

                for record in self.records

            )

        )

    # ==========================================================
    # Qualidade
    # ==========================================================

    def quality_distribution(self) -> dict[str, int]:

        return dict(

            Counter(

                record.signal.quality

                for record in self.records

            )

        )

    # ==========================================================
    # Recomendações
    # ==========================================================

    def recommendation_distribution(self):

        return dict(

            Counter(

                record.signal.recommendation

                for record in self.records

            )

        )

    # ==========================================================
    # Regimes
    # ==========================================================

    def regime_distribution(self):

        return dict(

            Counter(

                record.market.regime

                for record in self.records

            )

        )

    # ==========================================================
    # Direções
    # ==========================================================

    def direction_distribution(self):

        return dict(

            Counter(

                record.market.direction

                for record in self.records

            )

        )

    # ==========================================================
    # Score Institucional
    # ==========================================================

    def institutional_score_statistics(self):

        if not self.records:

            return {

                "min": 0,

                "max": 0,

                "average": 0,

            }

        scores = [

            r.institutional.score

            for r in self.records

        ]

        return {

            "min": min(scores),

            "max": max(scores),

            "average": round(

                sum(scores) / len(scores),

                2,

            ),

        }

    # ==========================================================
    # Confidence
    # ==========================================================

    def confidence_statistics(self):

        if not self.records:

            return {

                "min": 0,

                "max": 0,

                "average": 0,

            }

        values = [

            r.institutional.confidence

            for r in self.records

        ]

        return {

            "min": min(values),

            "max": max(values),

            "average": round(

                sum(values) / len(values),

                4,

            ),

        }

    # ==========================================================
    # Aprovação dos filtros
    # ==========================================================

    def filter_statistics(self):

        statistics = {}

        for record in self.records:

            for item in record.filters:

                name = item.name

                if name not in statistics:

                    statistics[name] = {

                        "approved": 0,

                        "rejected": 0,

                    }

                if item.approved:

                    statistics[name]["approved"] += 1

                else:

                    statistics[name]["rejected"] += 1

        return statistics

    # ==========================================================
    # Motivos de rejeição
    # ==========================================================

    def rejection_reasons(self):

        counter = Counter()

        for record in self.records:

            if not record.decision.approved:
                counter[record.decision.reason] += 1

        return dict(counter)

    # ==========================================================
    # Motivos de rejeição do TimeframeFilter
    # ==========================================================

    def timeframe_rejection_reasons(self):

        counter = Counter()

        for record in self.records:

            for item in record.filters:

                if (
                    item.name == "TimeframeFilter"
                    and not item.approved
                ):
                    counter[item.reason] += 1

        return dict(counter)

    # ==========================================================
    # Combinações do TimeframeFilter
    # ==========================================================

    def timeframe_combinations(self):

        counter = Counter()

        for record in self.records:

            for item in record.filters:

                if item.name != "TimeframeFilter":
                    continue

                metadata = item.metadata or {}

                trend_1d = metadata.get("trend_1d", "?")
                trend_4h = metadata.get("trend_4h", "?")
                trend_1h = metadata.get("trend_1h", "?")

                key = (
                    f"1d={trend_1d:<5} | "
                    f"4h={trend_4h:<5} | "
                    f"1h={trend_1h:<5}"
                )

                counter[key] += 1

        return dict(counter)