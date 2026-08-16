from logs.logger import setup_logger
from market_data.binance_rest import BinanceREST
from data.database import Database

import time


logger = setup_logger()


class MarketDataCollector:

    def __init__(self):

        self.exchange = BinanceREST()

        self.database = Database()

        self.database.create_tables()

        logger.info(
            "Market Data Collector iniciado"
        )


    def collect_price(self):

        try:

            price = self.exchange.get_btc_price()

            logger.info(
                f"BTCUSDT preco atual: {price}"
            )


            self.database.save_price(
                "BTCUSDT",
                price
            )


            logger.info(
                "Preco armazenado no banco"
            )


            return price


        except Exception as error:

            logger.error(
                f"Erro coleta mercado: {error}"
            )

            return None



    def start(self):

        logger.info(
            "Iniciando coleta continua"
        )


        while True:

            self.collect_price()

            time.sleep(10)



# =====================================================
# Compatibilidade com arquitetura anterior
# =====================================================

MarketCollector = MarketDataCollector



if __name__ == "__main__":


    collector = MarketDataCollector()

    collector.start()