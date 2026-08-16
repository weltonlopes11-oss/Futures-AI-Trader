import pandas as pd
import numpy as np

from logs.logger import setup_logger


logger = setup_logger()



class MarketRegimeDetector:



    def __init__(self, dataframe):

        self.df = dataframe.copy()



    def calculate(self):


        df = self.df.copy()


        logger.info(
            "Market Regime Detection v3 iniciado"
        )


        # =========================
        # RETURNS
        # =========================

        df["returns"] = (
            df["close"]
            .pct_change()
            *
            100
        )


        # =========================
        # VOLATILITY
        # =========================

        df["volatility"] = (
            df["returns"]
            .rolling(20)
            .std()
        )



        # =========================
        # ATR SIMPLIFICADO
        # =========================

        df["high_low"] = (
            df["high"]
            -
            df["low"]
        )


        df["atr"] = (
            df["high_low"]
            .rolling(14)
            .mean()
        )



        # =========================
        # EMA TREND
        # =========================

        df["ema50"] = (
            df["close"]
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
        )


        df["ema200"] = (
            df["close"]
            .ewm(
                span=200,
                adjust=False
            )
            .mean()
        )



        # =========================
        # EMA SLOPE
        # =========================

        df["ema_slope"] = (
            df["ema50"]
            -
            df["ema50"]
            .shift(10)
        )



        # =========================
        # ADX SIMPLIFICADO
        # =========================

        df["trend_strength"] = (
            abs(
                df["ema50"]
                -
                df["ema200"]
            )
            /
            df["close"]
            *
            100
        )



        # =========================
        # VOLATILITY THRESHOLDS
        # =========================


        low_vol = (
            df["volatility"]
            .quantile(.30)
        )


        high_vol = (
            df["volatility"]
            .quantile(.70)
        )



        regimes = []



        for _, row in df.iterrows():


            regime = "RANGE"



            volatility = row["volatility"]


            if pd.isna(volatility):

                regimes.append(
                    "RANGE"
                )

                continue



            # LOW VOLATILITY

            if volatility < low_vol:

                regime = (
                    "LOW_VOLATILITY"
                )



            # HIGH VOLATILITY

            elif volatility > high_vol:

                regime = (
                    "HIGH_VOLATILITY"
                )



            # TREND UP

            if (
                row["ema50"]
                >
                row["ema200"]

                and

                row["ema_slope"]
                >
                0

                and

                row["trend_strength"]
                >
                0.15
            ):

                regime = "TREND_UP"



            # TREND DOWN

            elif (
                row["ema50"]
                <
                row["ema200"]

                and

                row["ema_slope"]
                <
                0

                and

                row["trend_strength"]
                >
                0.15
            ):

                regime = "TREND_DOWN"



            regimes.append(
                regime
            )



        df["market_regime"] = regimes



        # =========================
        # SCORE
        # =========================


        score_map = {

            "TREND_UP":90,

            "TREND_DOWN":90,

            "HIGH_VOLATILITY":60,

            "RANGE":50,

            "LOW_VOLATILITY":40

        }



        df["regime_score"] = (
            df["market_regime"]
            .map(score_map)
            .fillna(50)
        )



        logger.info(
            "Market Regime Detection v3 concluído"
        )



        return df