from __future__ import annotations

import os
import traceback
from pathlib import Path

from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics
from backtest.simulator import BacktestSimulator

from filters.regime_filter import RegimeFilter
from filters.timeframe_filter import TimeframeFilter
from filters.trend_filter import TrendFilter
from filters.volatility_filter import VolatilityFilter

from telemetry.telemetry_report import TelemetryReport


BACKTEST_DATA_SOURCE = os.getenv(
    "BACKTEST_DATA_SOURCE",
    "LIVE",
).upper()

BACKTEST_FIXED_DATA_PATH = os.getenv(
    "BACKTEST_FIXED_DATA_PATH",
    str(Path("data") / "ETHUSDT_1m_baseline.csv"),
)

BACKTEST_SYMBOL = os.getenv(
    "BACKTEST_SYMBOL",
    "ETHUSDT",
)

BACKTEST_INTERVAL = os.getenv(
    "BACKTEST_INTERVAL",
    "1m",
)

BACKTEST_LIMIT = int(
    os.getenv("BACKTEST_LIMIT", "20000")
)

BACKTEST_WARMUP = int(
    os.getenv("BACKTEST_WARMUP", "200")
)


def build_filters():

    return [
        RegimeFilter(),
        TrendFilter(),
        VolatilityFilter(),
        TimeframeFilter(),
    ]


def print_report(metrics: BacktestMetrics):

    print()
    print("=" * 70)
    print("               FUTURES AI TRADER")
    print("                 BACKTEST REPORT")
    print("=" * 70)

    print()
    print(f"Candles analisados.....: {metrics.total_candles}")
    print(f"Operações aprovadas....: {metrics.approved_trades}")
    print(f"Operações rejeitadas...: {metrics.rejected_trades}")

    print()
    print(f"LONG...................: {metrics.long_trades}")
    print(f"SHORT..................: {metrics.short_trades}")
    print(f"NO TRADE...............: {metrics.no_trade}")

    print()
    print(f"Approval Rate..........: {metrics.approval_rate:.2f}%")
    print(f"Rejection Rate.........: {metrics.rejection_rate:.2f}%")
    print(f"Trading Rate...........: {metrics.traded_percentage:.2f}%")
    print(f"Ignored Rate...........: {metrics.ignored_percentage:.2f}%")

    print()

    print("-" * 70)
    print("QUALIDADE DOS SINAIS")
    print("-" * 70)

    for k, v in metrics.quality_distribution.items():
        print(f"{k:<25}{v}")

    print()

    print("-" * 70)
    print("REGIMES DE MERCADO")
    print("-" * 70)

    for k, v in metrics.market_regimes.items():
        print(f"{k:<25}{v}")

    print()

    print("-" * 70)
    print("DIREÇÕES")
    print("-" * 70)

    for k, v in metrics.market_directions.items():
        print(f"{k:<25}{v}")

    print()

    print("-" * 70)
    print("RECOMENDAÇÕES")
    print("-" * 70)

    for k, v in metrics.recommendations.items():
        print(f"{k:<25}{v}")

    print()
    print("=" * 70)
    print("BACKTEST FINALIZADO")
    print("=" * 70)


def main():

    print()
    print("=" * 70)
    print("Inicializando Backtest...")
    print("=" * 70)
    print()

    filters = build_filters()

    print(f"Filtros carregados: {len(filters)}")
    print(f"Fonte de dados.......: {BACKTEST_DATA_SOURCE}")

    if BACKTEST_DATA_SOURCE == "FIXED":
        print(f"Dataset fixo.........: {BACKTEST_FIXED_DATA_PATH}")

    engine = BacktestEngine(
        filters=filters,
        risk_validators=[],
        symbol=BACKTEST_SYMBOL,
        interval=BACKTEST_INTERVAL,
        data_source=BACKTEST_DATA_SOURCE,
        fixed_data_path=(
            BACKTEST_FIXED_DATA_PATH
            if BACKTEST_DATA_SOURCE == "FIXED"
            else None
        ),
    )

    print()
    print("Carregando histórico...")

    trades = engine.run(
        limit=BACKTEST_LIMIT,
        warmup=BACKTEST_WARMUP,
    )

    print()
    print("HISTÓRICO UTILIZADO")
    print(f"Candles carregados....: {engine.history_size}")
    print(f"Início................: {engine.history_start}")
    print(f"Fim...................: {engine.history_end}")
    print(f"Candles processados...: {len(trades)}")

    print()
    print("OPEN INTEREST")
    print(
        "Disponível............: "
        f"{'SIM' if engine.open_interest_available else 'NÃO'}"
    )

    if engine.open_interest_available:
        print(
            f"Cobertura..............: "
            f"{engine.open_interest_coverage:.2f}%"
        )
        print(
            f"Primeiro OI............: "
            f"{engine.open_interest_first}"
        )
        print(
            f"Último OI..............: "
            f"{engine.open_interest_last}"
        )

    simulator = BacktestSimulator()

    simulation = simulator.simulate(trades)

    metrics = BacktestMetrics(**simulation)

    print_report(metrics)

    print()
    print("=" * 70)
    print("GERANDO TELEMETRIA...")
    print("=" * 70)

    telemetry = TelemetryReport(
        engine.telemetry_storage.records
    )

    telemetry.print()

    print()

    print(
        f"Registros coletados: "
        f"{len(engine.telemetry_storage)}"
    )


if __name__ == "__main__":

    try:
        main()

    except Exception:
        print()
        print("=" * 70)
        print("ERRO DURANTE O BACKTEST")
        print("=" * 70)

        traceback.print_exc()

        print()
        print("=" * 70)
        print("FIM")
        print("=" * 70)
