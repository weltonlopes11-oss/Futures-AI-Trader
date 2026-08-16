from __future__ import annotations

import pandas as pd


class ATRCalculator:
    """
    Calcula o Average True Range (ATR).

    O ATR será utilizado por diversos módulos:

    - Volatility Detector
    - ATR Filter
    - Risk Manager
    - Adaptive Target
    - Dynamic Position Size
    - Stop Loss
    - Optimizer
    """

    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:

        if not isinstance(data, pd.DataFrame):
            raise TypeError("ATRCalculator espera um DataFrame.")

        if data.empty:
            raise ValueError("DataFrame vazio.")

        required = [
            "high",
            "low",
            "close",
        ]

        for column in required:
            if column not in data.columns:
                raise ValueError(
                    f"Coluna obrigatória ausente: {column}"
                )

        df = data.copy()

        previous_close = df["close"].shift(1)

        high_low = df["high"] - df["low"]

        high_close = (
            df["high"] - previous_close
        ).abs()

        low_close = (
            df["low"] - previous_close
        ).abs()

        df["true_range"] = pd.concat(
            [
                high_low,
                high_close,
                low_close,
            ],
            axis=1,
        ).max(axis=1)

        df["atr"] = (
            df["true_range"]
            .rolling(self.period)
            .mean()
        )

        return df

    def latest(self, data: pd.DataFrame) -> float:

        df = self.calculate(data)

        value = df.iloc[-1]["atr"]

        if pd.isna(value):
            return 0.0

        return float(value)