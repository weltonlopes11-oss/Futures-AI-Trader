from __future__ import annotations

from pathlib import Path

import pandas as pd

from logs.logger import setup_logger


logger = setup_logger()


class FixedHistoricalDataLoader:
    """
    Carrega candles de um arquivo CSV fixo para backtests reproduzíveis.
    """

    REQUIRED_COLUMNS = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self, limit: int | None = None) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset fixo não encontrado: {self.file_path}"
            )

        logger.info(
            f"Carregando dataset fixo: {self.file_path}"
        )

        df = pd.read_csv(self.file_path)

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Dataset fixo inválido. Colunas ausentes: {missing}"
            )

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_base",
            "taker_buy_quote",
        ]

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="raise",
                )

        df = (
            df.drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if limit is not None and limit > 0:
            df = df.tail(limit).reset_index(drop=True)

        logger.info(
            f"Candles fixos carregados: {len(df)}"
        )

        return df
