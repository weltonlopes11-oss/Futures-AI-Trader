import pandas as pd


class TrendIndicators:

    def __init__(self, df):

        self.df = df.copy()


    def calculate(self):

        self.df["ema_20"] = (
            self.df["close"]
            .ewm(span=20, adjust=False)
            .mean()
        )

        self.df["ema_50"] = (
            self.df["close"]
            .ewm(span=50, adjust=False)
            .mean()
        )

        self.df["ema_200"] = (
            self.df["close"]
            .ewm(span=200, adjust=False)
            .mean()
        )

        self.df["trend"] = 0


        self.df.loc[
            self.df["ema_20"] > self.df["ema_50"],
            "trend"
        ] = 1


        self.df.loc[
            self.df["ema_20"] < self.df["ema_50"],
            "trend"
        ] = -1


        return self.df


    def calculate_all(self):

        return self.calculate()