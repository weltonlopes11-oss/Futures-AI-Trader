from datetime import datetime

from database.market_repository import MarketRepository


class IndicatorStorage:
    """
    Responsável por transformar indicadores
    calculados em snapshots persistidos.
    """


    def __init__(self):

        self.repository = MarketRepository()



    def save(
        self,
        symbol,
        timeframe,
        dataframe
    ):
        """
        Recebe dataframe final dos indicadores
        e salva o último candle analisado.
        """


        if dataframe.empty:
            return False


        candle = dataframe.iloc[-1]



        snapshot = {

            "symbol": symbol,

            "timeframe": timeframe,

            "timestamp":
                str(candle.name),


            "close":
                candle.get("close"),



            # Momentum

            "rsi":
                candle.get("rsi"),

            "stoch_rsi":
                candle.get("stoch_rsi"),

            "roc":
                candle.get("roc"),



            # Trend

            "ema_fast":
                candle.get("ema_fast"),

            "ema_slow":
                candle.get("ema_slow"),

            "macd":
                candle.get("macd"),

            "adx":
                candle.get("adx"),



            # Volatility

            "atr":
                candle.get("atr"),

            "bb_high":
                candle.get("bb_high"),

            "bb_low":
                candle.get("bb_low"),



            # Volume

            "volume":
                candle.get("volume"),

            "relative_volume":
                candle.get("relative_volume"),

            "volume_spike":
                int(
                    candle.get(
                        "volume_spike",
                        False
                    )
                ),



            # Institutional

            "vwap":
                candle.get("vwap"),

            "mfi":
                candle.get("mfi"),

            "ad_line":
                candle.get("ad_line")

        }


        self.repository.save_snapshot(
            snapshot
        )


        return True