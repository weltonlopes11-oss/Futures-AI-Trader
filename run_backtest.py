from __future__ import annotations

import traceback

from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics
from backtest.simulator import BacktestSimulator

from filters.regime_filter import RegimeFilter
from filters.timeframe_filter import TimeframeFilter
from filters.trend_filter import TrendFilter
from filters.volatility_filter import VolatilityFilter

from telemetry.telemetry_report import TelemetryReport


# ==========================================================
# Registro oficial dos filtros institucionais
# ==========================================================

def build_filters():

    return [

        RegimeFilter(),

        TrendFilter(),

        VolatilityFilter(),

        TimeframeFilter(),

    ]


# ==========================================================
# Impressão do relatório principal
# ==========================================================

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


# ==========================================================
# Main
# ==========================================================

def main():

    print()
    print("=" * 70)
    print("Inicializando Backtest...")
    print("=" * 70)
    print()

    filters = build_filters()

    print(f"Filtros carregados: {len(filters)}")

    engine = BacktestEngine(

        filters=filters,

        risk_validators=[],

        symbol="ETHUSDT",

        interval="1m",

    )

    print()
    print("Carregando histórico...")

    trades = engine.run(

        limit=20000,

        warmup=200,

    )

    print(f"Candles processados: {len(trades)}")

    simulator = BacktestSimulator()

    simulation = simulator.simulate(trades)

    metrics = BacktestMetrics(**simulation)

    print_report(metrics)

    # ======================================================
    # TELEMETRIA
    # ======================================================

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


# ==========================================================
# Entry Point
# ==========================================================

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