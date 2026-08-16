import numpy as np
import pandas as pd


class TrendStrengthIndicators:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

    def calculate(self):

        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # ==========================================
        # MOVIMENTOS DIRECIONAIS
        # ==========================================

        up_move = high.diff()

        down_move = -low.diff()

        plus_dm = np.where(

            (up_move > down_move) &
            (up_move > 0),

            up_move,

            0.0

        )

        minus_dm = np.where(

            (down_move > up_move) &
            (down_move > 0),

            down_move,

            0.0

        )

        self.df["plus_dm"] = plus_dm

        self.df["minus_dm"] = minus_dm

        # ==========================================
        # TRUE RANGE
        # ==========================================

        previous_close = close.shift(1)

        tr1 = high - low

        tr2 = (high - previous_close).abs()

        tr3 = (low - previous_close).abs()

        tr = pd.concat(

            [

                tr1,

                tr2,

                tr3

            ],

            axis=1

        ).max(axis=1)

        self.df["tr_adx"] = tr

        # ==========================================
        # ATR (WILDER)
        # ==========================================

        atr = tr.rolling(14).mean()

        self.df["atr_adx"] = atr

        # ==========================================
        # +DI
        # ==========================================

        plus_di = (

            pd.Series(plus_dm)

            .rolling(14)

            .sum()

            /

            atr

        ) * 100

        self.df["plus_di"] = plus_di

        # ==========================================
        # -DI
        # ==========================================

        minus_di = (

            pd.Series(minus_dm)

            .rolling(14)

            .sum()

            /

            atr

        ) * 100

        self.df["minus_di"] = minus_di

        # ==========================================
        # DX
        # ==========================================

        dx = (

            (

                (

                    self.df["plus_di"]

                    -

                    self.df["minus_di"]

                ).abs()

            )

            /

            (

                self.df["plus_di"]

                +

                self.df["minus_di"]

            )

        ) * 100

        self.df["dx"] = dx

        # ==========================================
        # ADX
        # ==========================================

        self.df["adx"] = dx.rolling(14).mean()

        # ==========================================
        # SCORE
        # ==========================================

        score = []

        strength = []

        for value in self.df["adx"]:

            if pd.isna(value):

                score.append(0)

                strength.append("UNKNOWN")

            elif value < 20:

                score.append(25)

                strength.append("WEAK")

            elif value < 30:

                score.append(50)

                strength.append("NORMAL")

            elif value < 40:

                score.append(75)

                strength.append("STRONG")

            else:

                score.append(100)

                strength.append("EXTREME")

        self.df["trend_strength_score"] = score

        self.df["trend_strength"] = strength

        return self.df