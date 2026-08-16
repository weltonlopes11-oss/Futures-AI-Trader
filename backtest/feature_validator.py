import pandas as pd
import numpy as np

from logs.logger import setup_logger


logger = setup_logger()



class FeatureValidator:


    def __init__(self):

        logger.info(
            "Feature Validator iniciado"
        )



    def calculate_future_return(
        self,
        df,
        periods
    ):

        df = df.copy()

        df[f"future_{periods}"] = (
            df["close_1h"]
            .shift(-periods)
            /
            df["close_1h"]
            - 1
        ) * 100


        return df



    def validate_regime(
        self,
        df
    ):


        results = {}


        for regime in [
            "TREND_UP",
            "TREND_DOWN",
            "RANGE"
        ]:


            subset = df[
                df["market_regime"]
                ==
                regime
            ]


            if len(subset) == 0:

                continue



            avg_return = (
                subset["future_10"]
                .mean()
            )


            win_rate = (
                (
                    subset["future_10"] > 0
                )
                .sum()
                /
                len(subset)
            ) * 100



            results[regime] = {

                "samples":
                len(subset),

                "average_return":
                round(avg_return,4),

                "win_rate":
                round(win_rate,2)

            }


        return results



    def validate_momentum(
        self,
        df
    ):


        condition = (
            (df["rsi"] > 50)
            &
            (df["macd_hist"] > 0)
            &
            (df["momentum_score"] >= 60)
        )


        subset = df[
            condition
        ]


        if len(subset)==0:

            return {}



        return {

            "samples":
            len(subset),

            "win_rate":
            round(
                (
                    subset["future_10"] > 0
                )
                .sum()
                /
                len(subset)
                *
                100,
                2
            ),


            "average_return":
            round(
                subset["future_10"]
                .mean(),
                4
            )

        }



    def run(
        self,
        df
    ):


        logger.info(
            "Executando Feature Validation"
        )


        df = self.calculate_future_return(
            df,
            10
        )


        report = {


            "samples":
            len(df),


            "regime_analysis":
            self.validate_regime(df),


            "momentum_analysis":
            self.validate_momentum(df)

        }


        return report