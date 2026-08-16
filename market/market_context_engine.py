from __future__ import annotations

from market.market_context import MarketContext
from market.detectors.regime_detector import RegimeDetector


class MarketContextEngine:
    """
    Responsável por construir o contexto oficial do mercado.

    Toda a plataforma deve consumir apenas o MarketContext
    retornado por esta classe.
    """

    def __init__(
        self,
        regime_detector: RegimeDetector | None = None,
    ):

        self.regime_detector = (
            regime_detector or RegimeDetector()
        )

    def build(self, data):

        regime = self.regime_detector.detect(data)

        trend = regime.metadata["trend"]

        volatility = regime.metadata["volatility"]

        last = data.iloc[-1]

        return MarketContext(

            trend=trend.trend,

            volatility=volatility.volatility,

            regime=regime.regime,

            direction=regime.direction,

            confidence=regime.confidence,

            score=regime.score,

            metadata={

                "trend": trend,

                "volatility": volatility,

                "regime": regime,

                # Timeframes calculados pelo FeatureEngine
                "trend_1d": last.get("trend_1d"),
                "trend_4h": last.get("trend_4h"),
                "trend_1h": last.get("trend_1h"),

                # Open Interest histórico alinhado ao candle atual.
                # Nesta fase é apenas telemetria/contexto; não altera filtros.
                "open_interest": last.get("open_interest"),
                "open_interest_value": last.get("open_interest_value"),
                "open_interest_change_pct": last.get(
                    "open_interest_change_pct"
                ),

            },
        )
