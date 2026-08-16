import time

import pandas as pd

from market_data.collector import MarketDataCollector

from logs.logger import setup_logger


logger = setup_logger()


class HistoricalDataLoader:

    BINANCE_MAX_LIMIT = 1000

    def __init__(
        self,
        symbol="BTCUSDT",
        interval="1m",
    ):

        self.symbol = symbol

        self.interval = interval

        self.collector = MarketDataCollector()

    def load(
        self,
        limit=1000,
    ):

        logger.info(
            f"Carregando {limit} candles..."
        )

        candles = []

        remaining = limit

        end_time = None

        lote = 1

        while remaining > 0:

            request_size = min(
                remaining,
                self.BINANCE_MAX_LIMIT,
            )

            logger.info(
                f"Lote {lote} | "
                f"Quantidade: {request_size}"
            )

            batch = self.collector.exchange.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=request_size,
                end_time=end_time,
            )

            if not batch:

                logger.warning(
                    "Binance retornou lote vazio."
                )

                break

            candles = batch + candles

            oldest = batch[0][0]

            end_time = oldest - 1

            remaining -= len(batch)

            lote += 1

            time.sleep(0.15)

        logger.info(
            f"Total bruto de candles: {len(candles)}"
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

            "ignore",

        ]

        df = pd.DataFrame(
            candles,
            columns=columns,
        )

        numeric_columns = [

            "open",

            "high",

            "low",

            "close",

            "volume",

            "quote_volume",

            "taker_buy_base",

            "taker_buy_quote",

        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column]
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
        )

        df = df.drop_duplicates(
            subset="timestamp"
        )

        df = df.sort_values(
            "timestamp"
        ).reset_index(
            drop=True
        )

        logger.info(
            f"Candles finais: {len(df)}"
        )

        return df