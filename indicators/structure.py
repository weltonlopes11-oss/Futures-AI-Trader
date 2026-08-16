import numpy as np
import pandas as pd


class StructureIndicators:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

    def calculate(self):

        lookback = 10

        swing_highs = []
        swing_lows = []

        nearest_support = []
        nearest_resistance = []

        support_distance = []
        resistance_distance = []

        highs = self.df["high"].values
        lows = self.df["low"].values
        closes = self.df["close"].values

        for i in range(len(self.df)):

            if i < lookback:

                swing_highs.append(np.nan)
                swing_lows.append(np.nan)

                nearest_support.append(np.nan)
                nearest_resistance.append(np.nan)

                support_distance.append(np.nan)
                resistance_distance.append(np.nan)

                continue

            window_high = np.max(highs[i-lookback:i])
            window_low = np.min(lows[i-lookback:i])

            swing_highs.append(window_high)
            swing_lows.append(window_low)

            current_price = closes[i]

            nearest_support.append(window_low)
            nearest_resistance.append(window_high)

            support_distance.append(

                ((current_price - window_low) / current_price) * 100

            )

            resistance_distance.append(

                ((window_high - current_price) / current_price) * 100

            )

        self.df["swing_high"] = swing_highs
        self.df["swing_low"] = swing_lows

        self.df["nearest_support"] = nearest_support
        self.df["nearest_resistance"] = nearest_resistance

        self.df["support_distance_percent"] = support_distance
        self.df["resistance_distance_percent"] = resistance_distance

        return self.df