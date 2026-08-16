import time
import pandas as pd

from logs.logger import setup_logger


logger = setup_logger()



class BinanceHistoricalDownloader:


    def __init__(
        self,
        exchange
    ):

        self.exchange = exchange



    def download(
        self,
        symbol,
        interval,
        total_limit
    ):


        max_limit = 1500


        all_candles = []

        remaining = total_limit


        end_time = None



        while remaining > 0:


            request_limit = min(

                remaining,

                max_limit

            )



            candles = self.exchange.get_klines(

                symbol,

                interval,

                request_limit,

                end_time=end_time

            )



            if not candles:

                break



            all_candles.extend(

                candles

            )



            remaining -= len(candles)



            end_time = candles[0][0] - 1



            logger.info(

                f"Candles acumulados: {len(all_candles)}"

            )



            time.sleep(0.2)



        df = pd.DataFrame(

            all_candles

        )



        df = df.drop_duplicates()



        df = df.sort_values(

            by=0

        )



        return df