import pandas as pd


class IndicatorSnapshot:


    def create(
        self,
        dataframe
    ):


        if dataframe.empty:

            raise ValueError(
                "DataFrame de indicadores vazio"
            )


        last = dataframe.iloc[-1]


        snapshot = {


            "symbol": "BTCUSDT",


            "timestamp":
                str(last["timestamp"]),


            "close":
                float(last["close"]),



            "ema_20":
                float(last["ema_20"]),


            "ema_50":
                float(last["ema_50"]),


            "ema_200":
                float(last["ema_200"]),



            "trend":
                float(last["trend"]),



            "atr_14":
                float(last["atr_14"]),


            "volatility_percent":
                float(last["volatility_percent"]),



            "volume_sma_20":
                float(last["volume_sma_20"]),


            "volume_ratio":
                float(last["volume_ratio"]),


            "volume_spike":
                int(last["volume_spike"]),



            "rsi":
                float(last["rsi"]),



            "macd":
                float(last["macd"]),


            "macd_signal":
                float(last["macd_signal"]),


            "macd_hist":
                float(last["macd_hist"]),



            "stoch_rsi":
                float(last["stoch_rsi"]),



            "momentum_score":
                float(last["momentum_score"]),



            "institutional_candle":
                int(last["institutional_candle"]),



            "buy_pressure":
                float(last["buy_pressure"]),



            "institutional_score":
                float(last["institutional_score"])

        }


        return snapshot