from __future__ import annotations

from typing import Any

from filters.base_filter import BaseFilter
from filters.filter_result import FilterResult
from market.market_context import MarketContext


class TimeframeFilter(BaseFilter):
    """
    Valida o alinhamento entre os timeframes
    utilizando o MarketContext.
    """

    def evaluate(self, data: Any) -> FilterResult:

        if not isinstance(data, MarketContext):
            return self.reject("MarketContext inválido.")

        metadata = data.metadata

        trend_1d = str(
            metadata.get("trend_1d", "")
        ).upper()

        trend_4h = str(
            metadata.get("trend_4h", "")
        ).upper()

        trend_1h = str(
            metadata.get("trend_1h", "")
        ).upper()

        if not trend_1d:
            return self.reject("Trend 1D ausente.")

        if not trend_4h:
            return self.reject("Trend 4H ausente.")

        if not trend_1h:
            return self.reject("Trend 1H ausente.")

        trends = [
            trend_1d,
            trend_4h,
            trend_1h,
        ]

        bull = trends.count("BULL")
        bear = trends.count("BEAR")

        minimum_alignment = (
            self.config.get("minimum_alignment", 2)
        )

        require_daily_confirmation = (
            self.config.get(
                "require_daily_confirmation",
                True,
            )
        )

        metadata = {
            "trend_1d": trend_1d,
            "trend_4h": trend_4h,
            "trend_1h": trend_1h,
            "bull_count": bull,
            "bear_count": bear,
        }

        # -------------------------
        # LONG
        # -------------------------

        if bull >= minimum_alignment:

            if (
                require_daily_confirmation
                and trend_1d != "BULL"
            ):
                return self.reject(
                    "Tendência diária não confirma LONG.",
                    metadata,
                )

            score = 70 + (bull * 10)

            grade = self._grade(score)

            return self.approve(
                score=score,
                direction="LONG",
                grade=grade,
                metadata=metadata,
            )

        # -------------------------
        # SHORT
        # -------------------------

        if bear >= minimum_alignment:

            if (
                require_daily_confirmation
                and trend_1d != "BEAR"
            ):
                return self.reject(
                    "Tendência diária não confirma SHORT.",
                    metadata,
                )

            score = 70 + (bear * 10)

            grade = self._grade(score)

            return self.approve(
                score=score,
                direction="SHORT",
                grade=grade,
                metadata=metadata,
            )

        return self.reject(
            "Timeframes desalinhados.",
            metadata,
        )

    @staticmethod
    def _grade(score: float) -> str:

        if score >= 95:
            return "A+"

        if score >= 90:
            return "A"

        if score >= 80:
            return "B"

        return "C"