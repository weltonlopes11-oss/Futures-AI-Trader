from features.regime import MarketRegimeDetector

from logs.logger import setup_logger


logger = setup_logger()



class FeatureManager:


    def __init__(self, dataframe):

        self.df = dataframe.copy()



    def calculate(self):


        logger.info(
            "Feature Engineering iniciado"
        )


        regime_detector = MarketRegimeDetector(
            self.df
        )


        self.df = regime_detector.calculate()



        logger.info(
            "Feature Engineering concluído"
        )


        return self.df