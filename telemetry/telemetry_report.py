from __future__ import annotations

from telemetry.telemetry_statistics import TelemetryStatistics


class TelemetryReport:
    """
    Responsável apenas por apresentar
    as estatísticas da telemetria.

    Não realiza cálculos.

    Toda agregação é feita em
    TelemetryStatistics.
    """

    def __init__(self, records):

        self.statistics = TelemetryStatistics(records)

    # ==========================================================
    # Relatório completo
    # ==========================================================

    def print(self):

        print()

        print("=" * 70)
        print("INSTITUTIONAL TELEMETRY REPORT")
        print("=" * 70)

        self.print_summary()
        self.print_decisions()
        self.print_signal_quality()
        self.print_market()
        self.print_scores()
        self.print_filters()
        self.print_rejections()
        self.print_timeframe_rejections()
        self.print_filters()
        self.print_rejections()
        self.print_timeframe_combinations()

        print("=" * 70)
        print()

    # ==========================================================
    # Resumo
    # ==========================================================

    def print_summary(self):

        print()

        print("SUMMARY")

        print(f"Total records : {self.statistics.total_records}")

        print()

    # ==========================================================
    # Decisões
    # ==========================================================

    def print_decisions(self):

        print("DECISIONS")

        distribution = self.statistics.decision_distribution()

        for key, value in sorted(distribution.items()):

            print(f"{key:<15} {value}")

        print()

    # ==========================================================
    # Qualidade
    # ==========================================================

    def print_signal_quality(self):

        print("SIGNAL QUALITY")

        distribution = self.statistics.quality_distribution()

        for key, value in sorted(distribution.items()):

            print(f"{key:<15} {value}")

        print()

    # ==========================================================
    # Mercado
    # ==========================================================

    def print_market(self):

        print("MARKET")

        print()

        print("Directions")

        directions = self.statistics.direction_distribution()

        for key, value in sorted(directions.items()):

            print(f"{key:<15} {value}")

        print()

        print("Regimes")

        regimes = self.statistics.regime_distribution()

        for key, value in sorted(regimes.items()):

            print(f"{key:<15} {value}")

        print()

    # ==========================================================
    # Scores
    # ==========================================================

    def print_scores(self):

        score = self.statistics.institutional_score_statistics()

        confidence = self.statistics.confidence_statistics()

        print("INSTITUTIONAL SCORE")

        print(
            f"Average : {score['average']}"
        )

        print(
            f"Minimum : {score['min']}"
        )

        print(
            f"Maximum : {score['max']}"
        )

        print()

        print("CONFIDENCE")

        print(
            f"Average : {confidence['average']}"
        )

        print(
            f"Minimum : {confidence['min']}"
        )

        print(
            f"Maximum : {confidence['max']}"
        )

        print()

    # ==========================================================
    # Filtros
    # ==========================================================

    def print_filters(self):

        print("FILTER APPROVAL")

        filters = self.statistics.filter_statistics()

        for name in sorted(filters):

            approved = filters[name]["approved"]

            rejected = filters[name]["rejected"]

            total = approved + rejected

            if total == 0:

                rate = 0

            else:

                rate = approved / total * 100

            print(

                f"{name:<30}"

                f"Approved={approved:<6}"

                f"Rejected={rejected:<6}"

                f"Rate={rate:.1f}%"

            )

        print()

    # ==========================================================
    # Rejeições
    # ==========================================================

    def print_rejections(self):

        print("REJECTION REASONS")

        reasons = self.statistics.rejection_reasons()

        if not reasons:

            print("None")

            print()

            return

        for reason, qty in sorted(

            reasons.items(),

            key=lambda item: item[1],

            reverse=True,

        ):

            print(f"{reason:<40} {qty}")

        print()
        
    # ==========================================================
    # Timeframe Filter
    # ==========================================================

    def print_timeframe_rejections(self):

        print("TIMEFRAME FILTER REJECTION REASONS")

        reasons = self.statistics.timeframe_rejection_reasons()

        if not reasons:
            print("None")
            print()
            return

        for reason, qty in sorted(
            reasons.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{reason:<45} {qty}")

        print()

    # ==========================================================
    # Combinações dos Timeframes
    # ==========================================================

    def print_timeframe_combinations(self):

        print("TIMEFRAME COMBINATIONS")

        combinations = self.statistics.timeframe_combinations()

        if not combinations:
            print("None")
            print()
            return

        for combo, qty in sorted(
            combinations.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{combo:<45} {qty}")

        print()