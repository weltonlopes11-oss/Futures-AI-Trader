from __future__ import annotations

from pathlib import Path

import pandas as pd

from filters.institutional_filter_engine import InstitutionalFilterEngine
from market.market_context_engine import MarketContextEngine
from market.feature_engine import FeatureEngine
from backtest.historical_loader import HistoricalDataLoader
from backtest.fixed_historical_loader import FixedHistoricalDataLoader

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

    VALID_DATA_SOURCES = {
        "LIVE",
        "FIXED",
    }

    def __init__(
        self,
        filters: list,
        risk_validators: list | None = None,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        data_source: str = "LIVE",
        fixed_data_path: str | Path | None = None,
    ):

        self.symbol = symbol
        self.interval = interval
        self.data_source = data_source.upper()
        self.fixed_data_path = (
            Path(fixed_data_path)
            if fixed_data_path is not None
            else None
        )

        if self.data_source not in self.VALID_DATA_SOURCES:
            raise ValueError(
                "Fonte de dados inválida. Use LIVE ou FIXED."
            )

        if self.data_source == "FIXED":

            if self.fixed_data_path is None:
                raise ValueError(
                    "fixed_data_path é obrigatório quando data_source=FIXED."
                )

            self.loader = FixedHistoricalDataLoader(
                file_path=self.fixed_data_path,
            )

        else:

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

        self.history_start = None
        self.history_end = None
        self.history_size = 0

    def run(
        self,
        limit: int = 1000,
        warmup: int = 200,
    ) -> list:

        history = self.loader.load(limit)

        if history is None:
            raise RuntimeError(
                "Histórico vazio: loader não retornou candles."
            )

        if not isinstance(history, pd.DataFrame):
            raise TypeError(
                "Histórico inválido: loader deve retornar um DataFrame."
            )

        if history.empty:
            raise RuntimeError(
                "Histórico vazio: loader não retornou candles."
            )

        self.history_size = len(history)
        self.history_start = history.iloc[0]["timestamp"]
        self.history_end = history.iloc[-1]["timestamp"]

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
