import pandas as pd


class InstitutionalIndicators:


    def __init__(self, df):

        self.df = df.copy()



    def calculate(self):


        self.df["volume_average"] = (
            self.df["volume"]
            .rolling(20)
            .mean()
        )


        # Volume institucional

        self.df["volume_spike"] = (
            self.df["volume"]
            >
            self.df["volume_average"] * 2
        )


        # Candle institucional

        candle_size = (
            self.df["high"]
            -
            self.df["low"]
        )


        average_candle = (
            candle_size
            .rolling(20)
            .mean()
        )


        self.df["institutional_candle"] = (
            candle_size
            >
            average_candle * 1.8
        )



        # Pressão compradora/vendedora

        self.df["buy_pressure"] = (
            (
                self.df["close"]
                -
                self.df["low"]
            )
            /
            (
                self.df["high"]
                -
                self.df["low"]
            )
        )


        # Score institucional

        self.df["institutional_score"] = (
            self.calculate_score()
        )


        return self.df



    def calculate_score(self):

        score = pd.Series(
            50,
            index=self.df.index
        )


        score += (
            self.df["volume_spike"]
            .apply(
                lambda x:
                20 if x else 0
            )
        )


        score += (
            self.df["institutional_candle"]
            .apply(
                lambda x:
                20 if x else 0
            )
        )


        score += (
            self.df["buy_pressure"]
            .apply(
                lambda x:
                10 if x > 0.7
                else -10 if x < 0.3
                else 0
            )
        )


        return score.clip(0,100)