from indicators.trend import TrendIndicators
from indicators.volatility import VolatilityIndicators
from indicators.volume import VolumeIndicators
from indicators.momentum import MomentumIndicators
from indicators.institutional import InstitutionalIndicators
from indicators.structure import StructureIndicators
from indicators.trend_strength import TrendStrengthIndicators
from indicators.market_structure import MarketStructureIndicators

from intelligence.market_regime import MarketRegime


class IndicatorsEngine:


    def __init__(self, dataframe):

        if dataframe is None:

            raise ValueError(
                "DataFrame não informado."
            )


        self.df = dataframe.copy()



    def calculate(self):


        if self.df.empty:

            raise ValueError(
                "DataFrame vazio."
            )



        required_columns = [

            "open",

            "high",

            "low",

            "close",

            "volume"

        ]



        missing = [

            column

            for column in required_columns

            if column not in self.df.columns

        ]



        if missing:

            raise ValueError(

                f"Colunas obrigatórias ausentes: {missing}"

            )



        # ==========================================
        # TREND
        # ==========================================

        self.df = TrendIndicators(

            self.df

        ).calculate()



        # ==========================================
        # VOLATILITY
        # ==========================================

        self.df = VolatilityIndicators(

            self.df

        ).calculate()



        # ==========================================
        # VOLUME
        # ==========================================

        self.df = VolumeIndicators(

            self.df

        ).calculate()



        # ==========================================
        # MOMENTUM
        # ==========================================

        self.df = MomentumIndicators(

            self.df

        ).calculate()



        # ==========================================
        # INSTITUTIONAL
        # ==========================================

        self.df = InstitutionalIndicators(

            self.df

        ).calculate()



        # ==========================================
        # MARKET STRUCTURE BÁSICA
        # suporte/resistência
        # ==========================================

        self.df = StructureIndicators(

            self.df

        ).calculate()



        # ==========================================
        # TREND STRENGTH
        # ADX
        # ==========================================

        self.df = TrendStrengthIndicators(

            self.df

        ).calculate()



        # ==========================================
        # SMART MARKET STRUCTURE
        # BOS / CHOCH
        # HH / HL / LH / LL
        # ==========================================

        self.df = MarketStructureIndicators(

            self.df

        ).calculate()



        # ==========================================
        # MARKET REGIME
        # ==========================================

        self.df = MarketRegime(

            self.df

        ).calculate()



        return self.df