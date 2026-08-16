from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from backtest.open_interest_enricher import OpenInterestEnricher
from market_data.open_interest_client import OpenInterestClient


SYMBOL = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
SOURCE_FILE = Path(
    os.getenv(
        "BACKTEST_FIXED_DATA_PATH",
        str(Path("data") / "ETHUSDT_1m_baseline.csv"),
    )
)
OUTPUT_FILE = Path(
    os.getenv(
        "OPEN_INTEREST_BASELINE_PATH",
        str(Path("data") / "ETHUSDT_1m_baseline_oi.csv"),
    )
)
OI_PERIOD = os.getenv("OPEN_INTEREST_PERIOD", "5m")


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline de candles não encontrado: {SOURCE_FILE}"
        )

    candles = pd.read_csv(SOURCE_FILE)

    if "timestamp" not in candles.columns:
        raise ValueError("Baseline sem coluna timestamp.")

    candles["timestamp"] = pd.to_datetime(candles["timestamp"])
    candles = candles.sort_values("timestamp").reset_index(drop=True)

    start = candles["timestamp"].iloc[0]
    end = candles["timestamp"].iloc[-1]

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print()
    print("=" * 70)
    print("CAPTURA DE OPEN INTEREST")
    print("=" * 70)
    print(f"Símbolo...............: {SYMBOL}")
    print(f"Período OI............: {OI_PERIOD}")
    print(f"Início................: {start}")
    print(f"Fim...................: {end}")
    print(f"Baseline origem.......: {SOURCE_FILE}")

    client = OpenInterestClient()
    records = client.get_range(
        symbol=SYMBOL,
        start_time=start_ms,
        end_time=end_ms,
        period=OI_PERIOD,
    )

    oi = OpenInterestEnricher.from_binance_records(records)

    if oi.empty:
        raise RuntimeError("Binance não retornou histórico de Open Interest.")

    enriched = OpenInterestEnricher().enrich(candles, oi)

    coverage = enriched["open_interest"].notna().mean() * 100.0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_FILE, index=False)

    print(f"Registros OI..........: {len(oi)}")
    print(f"Cobertura nos candles.: {coverage:.2f}%")
    print(f"Arquivo gerado........: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
