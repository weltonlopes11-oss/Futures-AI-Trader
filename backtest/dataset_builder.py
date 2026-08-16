import os

import pandas as pd

from market_data.collector import MarketDataCollector

from logs.logger import setup_logger



logger = setup_logger()





class HistoricalDatasetBuilder:



    def __init__(

        self,
        symbol="BTCUSDT"

    ):


        self.symbol = symbol

        self.collector = MarketDataCollector()


        self.output_path = (

            "data/historical"

        )


        os.makedirs(

            self.output_path,

            exist_ok=True

        )






    def build(

        self,
        interval,
        limit

    ):


        logger.info(

            f"Criando dataset {self.symbol} {interval}"

        )



        candles = self.collector.exchange.get_klines(

            self.symbol,

            interval,

            limit

        )



        columns = [

            "timestamp",

            "open",

            "high",

            "low",

            "close",

            "volume",

            "close_time",

            "quote_volume",

            "trades",

            "taker_buy_base",

            "taker_buy_quote",

            "ignore"

        ]




        df = pd.DataFrame(

            candles,

            columns=columns

        )





        numeric_columns = [

            "open",

            "high",

            "low",

            "close",

            "volume",

            "quote_volume",

            "taker_buy_base",

            "taker_buy_quote"

        ]





        for column in numeric_columns:


            df[column] = pd.to_numeric(

                df[column]

            )





        df["timestamp"] = pd.to_datetime(

            df["timestamp"],

            unit="ms"

        )





        filename = (

            f"{self.output_path}/"

            f"{self.symbol}_"

            f"{interval}.csv"

        )





        df.to_csv(

            filename,

            index=False

        )





        logger.info(

            f"Dataset salvo: {filename}"

        )



        return df