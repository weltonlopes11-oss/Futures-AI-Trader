from __future__ import annotations

import pandas as pd


class OpenInterestEnricher:
    """Alinha observações históricas de Open Interest aos candles sem look-ahead."""

    REQUIRED_COLUMNS = {
        "timestamp",
        "sumOpenInterest",
        "sumOpenInterestValue",
    }

    @staticmethod
    def _normalize_timestamp(series: pd.Series) -> pd.Series:
        """
        Normaliza timestamps para a mesma resolução interna do pandas.

        merge_asof exige que as duas chaves temporais tenham exatamente
        o mesmo dtype. CSVs podem chegar como datetime64[us], enquanto
        timestamps convertidos a partir da Binance podem chegar como
        datetime64[ms]. Padronizamos ambos para datetime64[ns].
        """

        return pd.to_datetime(series).astype("datetime64[ns]")

    def enrich(
        self,
        candles: pd.DataFrame,
        open_interest: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(candles, pd.DataFrame):
            raise TypeError("candles deve ser um DataFrame.")

        if not isinstance(open_interest, pd.DataFrame):
            raise TypeError("open_interest deve ser um DataFrame.")

        if candles.empty:
            raise ValueError("Candles vazios.")

        if open_interest.empty:
            raise ValueError("Histórico de Open Interest vazio.")

        missing = self.REQUIRED_COLUMNS - set(open_interest.columns)
        if missing:
            raise ValueError(
                "Open Interest inválido. Colunas ausentes: "
                f"{sorted(missing)}"
            )

        left = candles.copy().reset_index(drop=True)
        left["timestamp"] = self._normalize_timestamp(left["timestamp"])
        left = left.sort_values("timestamp")

        right = open_interest.copy().reset_index(drop=True)
        right["timestamp"] = self._normalize_timestamp(right["timestamp"])
        right["open_interest"] = pd.to_numeric(
            right["sumOpenInterest"],
            errors="raise",
        )
        right["open_interest_value"] = pd.to_numeric(
            right["sumOpenInterestValue"],
            errors="raise",
        )

        right = (
            right[
                [
                    "timestamp",
                    "open_interest",
                    "open_interest_value",
                ]
            ]
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        right["open_interest_change_pct"] = (
            right["open_interest"].pct_change() * 100.0
        )

        merged = pd.merge_asof(
            left,
            right,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

        return merged

    @staticmethod
    def from_binance_records(records: list[dict]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "sumOpenInterest",
                    "sumOpenInterestValue",
                ]
            )

        df = pd.DataFrame(records)

        if "timestamp" not in df.columns:
            raise ValueError("Open Interest sem timestamp.")

        df["timestamp"] = pd.to_datetime(
            pd.to_numeric(df["timestamp"], errors="raise"),
            unit="ms",
        ).astype("datetime64[ns]")

        return df
