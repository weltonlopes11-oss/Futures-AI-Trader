import pandas as pd
import ta


class MomentumIndicators:

    def __init__(self, df):

        self.df = df.copy()


    def calculate(self):

        close = self.df["close"]


        # RSI
        self.df["rsi"] = (
            ta.momentum.RSIIndicator(
                close=close,
                window=14
            )
            .rsi()
        )


        # MACD

        macd = ta.trend.MACD(
            close=close
        )

        self.df["macd"] = macd.macd()

        self.df["macd_signal"] = (
            macd.macd_signal()
        )

        self.df["macd_hist"] = (
            macd.macd_diff()
        )


        # Stochastic RSI

        stoch = ta.momentum.StochRSIIndicator(
            close=close,
            window=14,
            smooth1=3,
            smooth2=3
        )

        self.df["stoch_rsi"] = (
            stoch.stochrsi()
        )


        # Score Momentum

        self.df["momentum_score"] = (
            self.calculate_score()
        )


        return self.df



    def calculate_score(self):

        score = pd.Series(
            50,
            index=self.df.index
        )


        # RSI

        score += (
            self.df["rsi"]
            .apply(
                lambda x:
                20 if x > 60
                else -20 if x < 40
                else 0
            )
        )


        # MACD

        score += (
            (
                self.df["macd"]
                >
                self.df["macd_signal"]
            )
            .apply(
                lambda x:
                15 if x else -15
            )
        )


        return score.clip(0,100)