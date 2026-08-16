from __future__ import annotations

from filters.institutional_filter_engine import InstitutionalFilterEngine
from market.market_context_engine import MarketContextEngine
from market.feature_engine import FeatureEngine
from backtest.historical_loader import HistoricalDataLoader

from decision.trade_decision_engine import TradeDecisionEngine
from decision.signal_score import SignalScoreCalculator
from decision.trade_setup import TradeSetupBuilder

from risk.risk_manager import RiskManager

from telemetry import (
    TelemetryEngine,
    TelemetryStorage,
)


class BacktestEngine:
    """
    Orquestrador oficial do backtest.

    Não possui regras de trading.

    Apenas conecta os componentes da arquitetura.
    """

    def __init__(
        self,
        filters: list,
        risk_validators: list | None = None,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
    ):

        self.symbol = symbol

        self.loader = HistoricalDataLoader(
            symbol=symbol,
            interval=interval,
        )

        self.feature_engine = FeatureEngine()

        self.market = MarketContextEngine()

        self.filters = InstitutionalFilterEngine(
            filters=filters,
        )

        self.decision = TradeDecisionEngine()

        self.signal = SignalScoreCalculator()

        self.setup_builder = TradeSetupBuilder()

        self.risk = RiskManager(
            validators=risk_validators,
        )

        # ==================================================
        # Telemetria
        # ==================================================

        self.telemetry_engine = TelemetryEngine(
            symbol=symbol,
        )

        self.telemetry_storage = TelemetryStorage()

    def run(
        self,
        limit: int = 1000,
        warmup: int = 200,
    ) -> list:

        history = self.loader.load(limit)

        history = self.feature_engine.enrich(history)

        trades = []

        self.telemetry_storage.clear()

        for index in range(warmup, len(history)):

            window = history.iloc[: index + 1]

            try:

                context = self.market.build(window)

                institutional_score = self.filters.evaluate(
                    context
                )

                decision = self.decision.evaluate(
                    context=context,
                    score=institutional_score,
                )

                signal = self.signal.calculate(
                    context=context,
                    decision=decision,
                )

                setup = self.setup_builder.build(
                    decision=decision,
                    signal=signal,
                )

                risk = self.risk.evaluate(
                    setup=setup,
                )

                trades.append(
                    {
                        "timestamp": history.iloc[index]["timestamp"],
                        "context": context,
                        "institutional_score": institutional_score,
                        "decision": decision,
                        "signal": signal,
                        "setup": setup,
                        "risk": risk,
                    }
                )

                # ==========================================
                # TELEMETRIA
                # ==========================================

                record = self.telemetry_engine.capture(
                    timestamp=history.iloc[index]["timestamp"],
                    close_price=float(
                        history.iloc[index]["close"]
                    ),
                    context=context,
                    institutional_score=institutional_score,
                    signal=signal,
                    decision=decision,
                    risk=risk,
                )

                self.telemetry_storage.add(record)

            except Exception as ex:

                print(
                    f"[Backtest] Candle {index}: {ex}"
                )

        return trades