import pandas as pd

from logs.logger import setup_logger


logger = setup_logger()



class RealFeatureValidator:


    def __init__(self):

        logger.info(
            "Real Feature Validator iniciado"
        )



    def load(
        self,
        path
    ):

        df = pd.read_csv(path)


        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )


        df = df.sort_values(
            "timestamp"
        )


        return df



    def add_future_return(
        self,
        df,
        periods=10
    ):


        df = df.copy()


        df["future_return"] = (
            df["close"]
            .shift(-periods)
            /
            df["close"]
            -1
        ) * 100


        return df



    def analyze_regimes(
        self,
        df
    ):


        result = {}


        regimes = (
            df["market_regime"]
            .unique()
        )


        for regime in regimes:


            subset = df[
                df["market_regime"]
                ==
                regime
            ]


            subset = subset.dropna(
                subset=["future_return"]
            )


            if len(subset) == 0:
                continue



            win_rate = (
                (
                    subset["future_return"]
                    >0
                )
                .sum()
                /
                len(subset)
            ) * 100



            result[regime] = {

                "samples":
                len(subset),


                "average_return":
                round(
                    subset["future_return"]
                    .mean(),
                    4
                ),


                "win_rate":
                round(
                    win_rate,
                    2
                )

            }



        return result



    def analyze_score(
        self,
        df
    ):


        high_score = df[
            df["regime_score"] >= 70
        ]



        if len(high_score)==0:

            return {}



        return {

            "samples":
            len(high_score),


            "average_return":
            round(
                high_score["future_return"]
                .mean(),
                4
            ),


            "win_rate":
            round(
                (
                    high_score["future_return"]
                    >0
                )
                .sum()
                /
                len(high_score)
                *
                100,
                2
            )

        }



    def run(
        self,
        path
    ):


        df = self.load(
            path
        )


        df = self.add_future_return(
            df
        )


        report = {

            "total_samples":
            len(df),


            "regime_distribution":
            df["market_regime"]
            .value_counts()
            .to_dict(),


            "regime_analysis":
            self.analyze_regimes(
                df
            ),


            "score_analysis":
            self.analyze_score(
                df
            )

        }


        return report