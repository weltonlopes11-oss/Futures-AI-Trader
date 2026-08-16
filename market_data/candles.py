from logs.logger import setup_logger
from market_data.binance_rest import BinanceREST
from data.database import Database

import pandas as pd


logger = setup_logger()



class CandleCollector:


    def __init__(self):

        self.exchange = BinanceREST()

        self.database = Database()

        self.database.create_tables()


        logger.info(
            "Candle Collector iniciado"
        )



    def collect_candles(
        self,
        symbol="BTCUSDT",
        interval="1m",
        limit=200
    ):


        try:

            candles = (
                self.exchange.get_klines(
                    symbol,
                    interval,
                    limit
                )
            )


            df = pd.DataFrame(
                candles,
                columns=[
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
            )


            columns = [
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]


            df[columns] = df[columns].astype(float)


            logger.info(
                f"{len(df)} candles carregados"
            )


            return df


        except Exception as error:


            logger.error(
                f"Erro candles: {error}"
            )


            return None