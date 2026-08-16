from __future__ import annotations

import os
from pathlib import Path

from backtest.historical_loader import HistoricalDataLoader


SYMBOL = os.getenv("BACKTEST_SYMBOL", "ETHUSDT")
INTERVAL = os.getenv("BACKTEST_INTERVAL", "1m")
LIMIT = int(os.getenv("BACKTEST_LIMIT", "20000"))
OUTPUT = Path(
    os.getenv(
        "BACKTEST_FIXED_DATA_PATH",
        str(Path("data") / "ETHUSDT_1m_baseline.csv"),
    )
)


def main():
    print()
    print("=" * 70)
    print("CAPTURANDO BASELINE HISTÓRICO")
    print("=" * 70)
    print(f"Símbolo...............: {SYMBOL}")
    print(f"Intervalo.............: {INTERVAL}")
    print(f"Candles solicitados...: {LIMIT}")
    print(f"Destino...............: {OUTPUT}")
    print()

    loader = HistoricalDataLoader(
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    history = loader.load(LIMIT)

    if history is None or history.empty:
        raise RuntimeError(
            "Não foi possível capturar o baseline: histórico vazio."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("BASELINE CRIADO")
    print(f"Candles...............: {len(history)}")
    print(f"Início................: {history.iloc[0]['timestamp']}")
    print(f"Fim...................: {history.iloc[-1]['timestamp']}")
    print(f"Arquivo...............: {OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
