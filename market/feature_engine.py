from __future__ import annotations

import pandas as pd


class FeatureEngine:
    """
    Responsável por enriquecer o DataFrame com informações
    derivadas antes da análise de mercado.

    V1:
        - trend_1h
        - trend_4h
        - trend_1d
    """

    EMA_CONFIG = {
        "1m": (20, 50),
        "1h": (21, 55),
        "4h": (8, 34),
        "1d": (5, 21),
    }

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:

        self._validate_dataframe(df)

        # elimina qualquer índice antigo para evitar ambiguidades
        result = df.copy().reset_index(drop=True)

        result = self._merge_timeframe(
            result,
            timeframe="1h",
            column_name="trend_1h",
        )

        result = self._merge_timeframe(
            result,
            timeframe="4h",
            column_name="trend_4h",
        )

        result = self._merge_timeframe(
            result,
            timeframe="1d",
            column_name="trend_1d",
        )

        return result

    # ---------------------------------------------------------

    def _validate_dataframe(self, df: pd.DataFrame):

        required = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"Colunas ausentes: {missing}"
            )

    # ---------------------------------------------------------

    def _merge_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        column_name: str,
    ) -> pd.DataFrame:

        # garante que não exista um índice chamado timestamp
        base = df.copy().reset_index(drop=True)

        base["timestamp"] = pd.to_datetime(base["timestamp"])

        base = base.set_index("timestamp")

        candles = (
            base.resample(timeframe)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        if timeframe not in self.EMA_CONFIG:
            raise ValueError(
                f"Timeframe não configurado: {timeframe}"
            )

        fast_period, slow_period = self.EMA_CONFIG[timeframe]

        candles[column_name] = self._calculate_trend(
            candles,
            fast_period,
            slow_period,
        )

        trend = candles[[column_name]]

        # evita ambiguidade entre índice e coluna
        left = df.copy().reset_index(drop=True)
        left["timestamp"] = pd.to_datetime(left["timestamp"])
        left = left.sort_values("timestamp")

        merged = pd.merge_asof(
            left,
            trend.sort_index(),
            left_on="timestamp",
            right_index=True,
            direction="backward",
        )

        return merged

    # ---------------------------------------------------------

    def _calculate_trend(
        self,
        candles: pd.DataFrame,
        fast_period: int,
        slow_period: int,
    ) -> pd.Series:

        ema_fast = candles["close"].ewm(
            span=fast_period,
            adjust=False,
        ).mean()

        ema_slow = candles["close"].ewm(
            span=slow_period,
            adjust=False,
        ).mean()

        trend = pd.Series(
            index=candles.index,
            dtype="object",
        )

        trend.loc[ema_fast > ema_slow] = "BULL"
        trend.loc[ema_fast < ema_slow] = "BEAR"

        trend = trend.fillna("RANGE")

        return trend