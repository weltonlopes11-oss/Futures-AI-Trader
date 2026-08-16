import pandas as pd
import numpy as np

from logs.logger import setup_logger


logger = setup_logger()



class RegimeCalibrator:


    def __init__(self):

        logger.info(
            "Regime Calibrator iniciado"
        )



    def load(
        self,
        path
    ):

        df = pd.read_csv(path)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        return df



    def calculate_metrics(
        self,
        df
    ):


        df = df.copy()


        df["returns"] = (
            df["close"]
            .pct_change()
            *
            100
        )


        df["candle_range"] = (
            (
                df["close"]
                -
                df["close"].shift(1)
            )
            .abs()
            /
            df["close"].shift(1)
            *
            100
        )



        metrics = {


            "samples":
            len(df),


            "return_mean":
            round(
                df["returns"]
                .mean(),
                5
            ),


            "return_std":
            round(
                df["returns"]
                .std(),
                5
            ),


            "volatility_mean":
            round(
                df["candle_range"]
                .mean(),
                5
            ),


            "volatility_max":
            round(
                df["candle_range"]
                .max(),
                5
            ),



            "percentile_50":
            round(
                df["candle_range"]
                .quantile(.50),
                5
            ),


            "percentile_90":
            round(
                df["candle_range"]
                .quantile(.90),
                5
            )

        }


        return metrics



    def regime_distribution(
        self,
        df
    ):


        return (
            df["market_regime"]
            .value_counts()
            .to_dict()
        )



    def suggest_thresholds(
        self,
        df
    ):


        volatility = (
            df["close"]
            .pct_change()
            .abs()
            *
            100
        )



        low = round(
            volatility.quantile(.30),
            4
        )


        high = round(
            volatility.quantile(.70),
            4
        )



        return {


            "suggested_low_volatility":
            low,


            "suggested_high_volatility":
            high


        }



    def run(
        self,
        path
    ):


        df = self.load(
            path
        )


        report = {


            "metrics":
            self.calculate_metrics(
                df
            ),


            "current_distribution":
            self.regime_distribution(
                df
            ),


            "suggested_thresholds":
            self.suggest_thresholds(
                df
            )


        }


        return report