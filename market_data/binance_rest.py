import time
import requests

from logs.logger import setup_logger

logger = setup_logger()


class BinanceREST:

    BASE_URL = "https://fapi.binance.com"

    MAX_LIMIT = 1000

    def __init__(self):

        logger.info(
            "Binance REST inicializado"
        )

    def get_btc_price(self):

        try:

            url = (
                f"{self.BASE_URL}"
                "/fapi/v1/ticker/price"
            )

            params = {
                "symbol": "BTCUSDT"
            }

            response = requests.get(
                url,
                params=params,
                timeout=10
            )

            response.raise_for_status()

            data = response.json()

            return float(data["price"])

        except Exception as error:

            logger.error(
                f"Erro BTC price: {error}"
            )

            return None

    def get_klines(
        self,
        symbol="BTCUSDT",
        interval="1m",
        limit=1000,
        end_time=None,
    ):

        try:

            url = (
                f"{self.BASE_URL}"
                "/fapi/v1/klines"
            )

            params = {

                "symbol": symbol,

                "interval": interval,

                "limit": min(limit, self.MAX_LIMIT),

            }

            if end_time is not None:

                params["endTime"] = int(end_time)

            response = requests.get(
                url,
                params=params,
                timeout=20,
            )

            response.raise_for_status()

            candles = response.json()

            logger.info(
                f"Klines recebidos: {len(candles)}"
            )

            return candles

        except Exception as error:

            logger.error(
                f"Erro Binance Klines: {error}"
            )

            return []