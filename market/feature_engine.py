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

    Regra de causalidade:
        candles de timeframes superiores só ficam disponíveis
        no instante em que o período agregado já foi encerrado.
    """

    EMA_CONFIG = {
        "1m": (20, 50),
        "1h": (21, 55),
        "4h": (8, 34),
        "1D": (5, 21),
    }

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recebe candles de 1 minuto e devolve o mesmo DataFrame
        enriquecido com tendências de múltiplos timeframes.
        """

        self._validate_dataframe(df)

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
            timeframe="1D",
            column_name="trend_1d",
        )

        return result

    # ---------------------------------------------------------

    def _validate_dataframe(self, df: pd.DataFrame):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("FeatureEngine espera um DataFrame.")

        if df.empty:
            raise ValueError("DataFrame vazio.")

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

        if timeframe not in self.EMA_CONFIG:
            raise ValueError(
                f"Timeframe não configurado: {timeframe}"
            )

        base = df.copy().reset_index(drop=True)
        base["timestamp"] = pd.to_datetime(base["timestamp"])
        base = base.sort_values("timestamp")
        base = base.set_index("timestamp")

        # closed="left" cria intervalos [inicio, fim).
        # label="right" rotula a vela no primeiro instante em que
        # todos os dados daquele período já estão disponíveis.
        # Ex.: a vela 1H [10:00, 11:00) recebe timestamp 11:00.
        candles = (
            base.resample(
                timeframe,
                closed="left",
                label="right",
            )
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

        fast_period, slow_period = self.EMA_CONFIG[timeframe]

        candles[column_name] = self._calculate_trend(
            candles,
            fast_period,
            slow_period,
        )

        trend = candles[[column_name]].sort_index()

        left = df.copy().reset_index(drop=True)
        left["timestamp"] = pd.to_datetime(left["timestamp"])
        left = left.sort_values("timestamp")

        # backward garante que cada candle-base só veja o último
        # timeframe superior cujo timestamp de disponibilidade seja
        # menor ou igual ao timestamp atual.
        merged = pd.merge_asof(
            left,
            trend,
            left_on="timestamp",
            right_index=True,
            direction="backward",
            allow_exact_matches=True,
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
