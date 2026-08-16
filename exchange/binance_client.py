from binance.client import Client
from loguru import logger


class BinanceClient:

    def __init__(self):

        self.client = Client()

        logger.info("Cliente Binance inicializado")


    def get_price(self, symbol="BTCUSDT"):

        ticker = self.client.get_symbol_ticker(symbol=symbol)

        price = float(ticker["price"])

        logger.info(f"{symbol}: {price}")

        return price