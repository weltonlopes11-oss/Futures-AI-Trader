import pandas as pd

from features.manager import FeatureManager
from logs.logger import setup_logger


logger = setup_logger()



class RealFeatureInjector:


    def __init__(self):

        logger.info(
            "Real Feature Injector iniciado"
        )



    def load_dataset(
        self,
        path
    ):


        df = pd.read_csv(
            path
        )


        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )


        logger.info(
            f"Candles carregados: {len(df)}"
        )


        return df



    def validate_ohlcv(
        self,
        df
    ):


        required = [

            "timestamp",

            "open",
            "high",
            "low",
            "close",
            "volume"

        ]


        missing = [

            col
            for col in required
            if col not in df.columns

        ]


        if missing:

            raise Exception(
                f"Dataset sem colunas OHLCV: {missing}"
            )



        return True



    def add_multi_timeframe_features(
        self,
        df
    ):


        df = df.copy()


        # ==========================
        # 4H FEATURES
        # ==========================


        df["close_4h"] = (
            df["close"]
            .rolling(
                4
            )
            .mean()
        )


        df["volume_4h"] = (
            df["volume"]
            .rolling(
                4
            )
            .sum()
        )



        # ==========================
        # 1D FEATURES
        # ==========================


        df["close_1d"] = (
            df["close"]
            .rolling(
                24
            )
            .mean()
        )


        df["volume_1d"] = (
            df["volume"]
            .rolling(
                24
            )
            .sum()
        )



        return df



    def apply_feature_engineering(
        self,
        df
    ):


        logger.info(
            "Aplicando Feature Engineering real"
        )


        # mantém OHLCV

        self.validate_ohlcv(
            df
        )


        # adiciona multi timeframe

        df = self.add_multi_timeframe_features(
            df
        )


        # Feature Manager

        manager = FeatureManager(
            df
        )


        df = manager.calculate()



        return df



    def save(
        self,
        df,
        path
    ):


        df.to_csv(
            path,
            index=False
        )


        logger.info(
            f"Features salvas: {path}"
        )



    def run(
        self,
        input_path,
        output_path
    ):


        df = self.load_dataset(
            input_path
        )


        df = self.apply_feature_engineering(
            df
        )


        self.save(
            df,
            output_path
        )


        return df