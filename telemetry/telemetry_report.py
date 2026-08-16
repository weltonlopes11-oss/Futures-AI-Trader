from __future__ import annotations

from telemetry.open_interest_statistics import OpenInterestStatistics
from telemetry.telemetry_statistics import TelemetryStatistics


class TelemetryReport:
    """Apresenta as estatísticas agregadas da telemetria."""

    def __init__(self, records):
        self.statistics = TelemetryStatistics(records)
        self.open_interest = OpenInterestStatistics(records)

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
        self.print_timeframe_combinations()
        self.print_open_interest_analysis()

        print("=" * 70)
        print()

    def print_summary(self):
        print()
        print("SUMMARY")
        print(f"Total records : {self.statistics.total_records}")
        print()

    def print_decisions(self):
        print("DECISIONS")
        distribution = self.statistics.decision_distribution()
        for key, value in sorted(distribution.items()):
            print(f"{key:<15} {value}")
        print()

    def print_signal_quality(self):
        print("SIGNAL QUALITY")
        distribution = self.statistics.quality_distribution()
        for key, value in sorted(distribution.items()):
            print(f"{key:<15} {value}")
        print()

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

    def print_scores(self):
        score = self.statistics.institutional_score_statistics()
        confidence = self.statistics.confidence_statistics()

        print("INSTITUTIONAL SCORE")
        print(f"Average : {score['average']}")
        print(f"Minimum : {score['min']}")
        print(f"Maximum : {score['max']}")
        print()

        print("CONFIDENCE")
        print(f"Average : {confidence['average']}")
        print(f"Minimum : {confidence['min']}")
        print(f"Maximum : {confidence['max']}")
        print()

    def print_filters(self):
        print("FILTER APPROVAL")
        filters = self.statistics.filter_statistics()

        for name in sorted(filters):
            approved = filters[name]["approved"]
            rejected = filters[name]["rejected"]
            total = approved + rejected
            rate = approved / total * 100 if total else 0

            print(
                f"{name:<30}"
                f"Approved={approved:<6}"
                f"Rejected={rejected:<6}"
                f"Rate={rate:.1f}%"
            )
        print()

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

    def print_open_interest_analysis(self):
        print("OPEN INTEREST STATISTICAL ANALYSIS")

        if self.open_interest.total_observations == 0:
            print("No Open Interest observations")
            print()
            return

        print(f"Real OI observations : {self.open_interest.total_observations}")
        print()

        print("OI Direction")
        for key, value in sorted(
            self.open_interest.oi_direction_distribution().items()
        ):
            print(f"{key:<15} {value}")

        print()
        print("OI x Decision")
        for decision, distribution in sorted(
            self.open_interest.by_decision().items()
        ):
            print(
                f"{decision:<15} "
                f"UP={distribution.get('UP', 0):<6} "
                f"DOWN={distribution.get('DOWN', 0):<6} "
                f"STABLE={distribution.get('STABLE', 0):<6}"
            )

        print()
        print("OI x Market Regime")
        for regime, distribution in sorted(
            self.open_interest.by_regime().items()
        ):
            print(
                f"{regime:<20} "
                f"UP={distribution.get('UP', 0):<6} "
                f"DOWN={distribution.get('DOWN', 0):<6} "
                f"STABLE={distribution.get('STABLE', 0):<6}"
            )

        print()
        print("Price x OI")
        combinations = self.open_interest.price_oi_combinations()
        for key, value in sorted(
            combinations.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{key:<30} {value}")

        print()
        print("OI Change Distribution (%)")
        stats = self.open_interest.change_statistics()["oi"]
        print(f"Count   : {stats['count']}")
        print(f"Average : {stats['average']}")
        print(f"Median  : {stats['median']}")
        print(f"Minimum : {stats['min']}")
        print(f"P10     : {stats['p10']}")
        print(f"P25     : {stats['p25']}")
        print(f"P75     : {stats['p75']}")
        print(f"P90     : {stats['p90']}")
        print(f"Maximum : {stats['max']}")

        print()
        print("OI Change by Decision (%)")
        for decision, values in sorted(
            self.open_interest.decision_change_statistics().items()
        ):
            print(
                f"{decision:<15} "
                f"n={values['count']:<5} "
                f"avg={values['average']:<10} "
                f"med={values['median']:<10} "
                f"p25={values['p25']:<10} "
                f"p75={values['p75']:<10}"
            )

        print()
