import pandas as pd


class VolumeIndicators:

    def __init__(self, df):

        self.df = df.copy()


    def calculate(self):

        # Volume médio
        self.df["volume_sma_20"] = (
            self.df["volume"]
            .rolling(window=20)
            .mean()
        )


        # Relação volume atual / média
        self.df["volume_ratio"] = (
            self.df["volume"] /
            self.df["volume_sma_20"]
        )


        # Spike de volume institucional
        self.df["volume_spike"] = 0


        self.df.loc[
            self.df["volume_ratio"] >= 2,
            "volume_spike"
        ] = 1


        return self.df


    def calculate_all(self):

        return self.calculate()