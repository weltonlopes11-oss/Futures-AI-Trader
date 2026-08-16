import numpy as np


class MarketRegime:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

    def calculate(self):

        regimes = []

        for _, row in self.df.iterrows():

            regime = "UNKNOWN"

            ema20 = row["ema_20"]
            ema50 = row["ema_50"]
            ema200 = row["ema_200"]

            atr = row["atr_14"]
            volatility = row["volatility_percent"]

            trend = row["trend"]

            # -------------------------
            # Forte tendência de alta
            # -------------------------

            if (
                ema20 > ema50 > ema200
                and trend == 1
                and volatility > 0.015
            ):

                regime = "UPTREND"

            # -------------------------
            # Forte tendência de baixa
            # -------------------------

            elif (
                ema20 < ema50 < ema200
                and trend == -1
                and volatility > 0.015
            ):

                regime = "DOWNTREND"

            # -------------------------
            # Alta volatilidade
            # -------------------------

            elif volatility > 0.035:

                regime = "HIGH_VOLATILITY"

            # -------------------------
            # Mercado lateral
            # -------------------------

            else:

                regime = "RANGE"

            regimes.append(regime)

        self.df["market_regime"] = regimes

        return self.df