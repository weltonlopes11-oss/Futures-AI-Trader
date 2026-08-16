import pandas as pd
import os

from logs.logger import setup_logger


logger = setup_logger()



class MultiTimeframeDataset:


    def __init__(self):

        logger.info(
            "MultiTimeframe Dataset iniciado"
        )


    def load_csv(
        self,
        path
    ):

        df = pd.read_csv(path)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        )

        return df



    def merge(
        self,
        symbol
    ):


        base_path = "data/historical"


        df_1h = self.load_csv(
            f"{base_path}/{symbol}_1h.csv"
        )


        df_4h = self.load_csv(
            f"{base_path}/{symbol}_4h.csv"
        )


        df_1d = self.load_csv(
            f"{base_path}/{symbol}_1d.csv"
        )



        # Renomear colunas


        df_1h = df_1h.rename(
            columns={
                "close":
                "close_1h",

                "volume":
                "volume_1h"
            }
        )


        df_4h = df_4h.rename(
            columns={
                "close":
                "close_4h",

                "volume":
                "volume_4h"
            }
        )


        df_1d = df_1d.rename(
            columns={
                "close":
                "close_1d",

                "volume":
                "volume_1d"
            }
        )



        # Selecionar somente necessário


        df_1h = df_1h[
            [
                "timestamp",
                "close_1h",
                "volume_1h"
            ]
        ]


        df_4h = df_4h[
            [
                "timestamp",
                "close_4h",
                "volume_4h"
            ]
        ]


        df_1d = df_1d[
            [
                "timestamp",
                "close_1d",
                "volume_1d"
            ]
        ]



        # Merge temporal


        merged = pd.merge_asof(
            df_1h,
            df_4h,
            on="timestamp",
            direction="backward"
        )


        merged = pd.merge_asof(
            merged,
            df_1d,
            on="timestamp",
            direction="backward"
        )



        merged = merged.dropna()



        output = (
            f"{base_path}/"
            f"{symbol}_multiframe.csv"
        )


        merged.to_csv(
            output,
            index=False
        )


        logger.info(
            f"Dataset multi timeframe salvo: {output}"
        )


        return merged