import pandas as pd


class VolatilityIndicators:

    def __init__(self, df):

        self.df = df.copy()


    def calculate(self):

        # True Range
        self.df["previous_close"] = (
            self.df["close"].shift(1)
        )


        self.df["tr"] = self.df.apply(
            lambda row: max(
                row["high"] - row["low"],
                abs(row["high"] - row["previous_close"])
                if pd.notna(row["previous_close"])
                else 0,
                abs(row["low"] - row["previous_close"])
                if pd.notna(row["previous_close"])
                else 0,
            ),
            axis=1
        )


        # ATR 14
        self.df["atr_14"] = (
            self.df["tr"]
            .rolling(window=14)
            .mean()
        )


        # Volatilidade percentual
        self.df["volatility_percent"] = (
            self.df["atr_14"] /
            self.df["close"]
        ) * 100


        return self.df


    def calculate_all(self):

        return self.calculate()