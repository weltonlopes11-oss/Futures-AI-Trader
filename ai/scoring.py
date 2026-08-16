import pandas as pd


class ScoreEngine:


    def calculate(self, df):

        score = pd.Series(
            50,
            index=df.index
        )


        # Trend (25%)

        score += (
            (df["trend_score"] - 50)
            * 0.25
        )


        # Momentum (20%)

        score += (
            (df["momentum_score"] - 50)
            * 0.20
        )


        # Volume (15%)

        score += (
            (df["volume_score"] - 50)
            * 0.15
        )


        # Institutional (25%)

        score += (
            (df["institutional_score"] - 50)
            * 0.25
        )


        # Volatility (15%)

        score += (
            (df["volatility_score"] - 50)
            * 0.15
        )


        return score.clip(0,100)