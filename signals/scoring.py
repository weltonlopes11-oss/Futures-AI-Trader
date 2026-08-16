from logs.logger import setup_logger


logger = setup_logger()



class SignalScoringEngine:


    def __init__(
        self,
        snapshot,
        analysis=None
    ):

        self.snapshot = snapshot
        self.analysis = analysis or {}



    def calculate(self):


        long_score = 0
        short_score = 0


        reasons = []
        warnings = []



        long_score += self.long_trend(reasons)

        short_score += self.short_trend(reasons)



        long_score += self.long_momentum(reasons)

        short_score += self.short_momentum(reasons)



        long_score += self.long_structure(reasons)

        short_score += self.short_structure(reasons)



        long_score += self.long_regime(reasons)

        short_score += self.short_regime(reasons)



        long_score += self.long_volume(reasons)

        short_score += self.short_volume(reasons)



        penalties = self.calculate_penalties(
            warnings
        )


        long_score -= penalties

        short_score -= penalties



        long_score = max(
            0,
            min(long_score,100)
        )


        short_score = max(
            0,
            min(short_score,100)
        )



        if long_score > short_score:

            direction = "LONG"

            final_score = long_score


        elif short_score > long_score:

            direction = "SHORT"

            final_score = short_score


        else:

            direction = "NEUTRAL"

            final_score = 50



        result = {


            "direction": direction,


            "signal": self.classify(
                final_score
            ),


            "long_score": long_score,


            "short_score": short_score,


            "confidence": self.calculate_confidence(
                final_score
            ),


            "risk": self.risk_level(
                final_score
            ),


            "reasons": reasons,


            "warnings": warnings


        }



        logger.info(
            f"Directional Signal Score: {result}"
        )


        return result





    # ============================
    # LONG
    # ============================


    def long_trend(self,reasons):


        score = 0


        if self.snapshot.get(
            "trend",
            0
        ) > 0:


            score += 15

            reasons.append(
                "Tendência bullish"
            )



        if self.snapshot.get(
            "adx",
            0
        ) >= 25:


            score += 10

            reasons.append(
                "ADX confirma força"
            )


        return score




    def short_trend(self,reasons):


        score = 0


        if self.snapshot.get(
            "trend",
            0
        ) < 0:


            score += 15

            reasons.append(
                "Tendência bearish"
            )



        if self.snapshot.get(
            "adx",
            0
        ) >= 25:


            score += 10


        return score




    # ============================
    # MOMENTUM
    # ============================


    def long_momentum(self,reasons):


        score = 0


        if self.snapshot.get(
            "rsi",
            50
        ) > 50:


            score += 7



        if self.snapshot.get(
            "macd_hist",
            0
        ) > 0:


            score += 6

            reasons.append(
                "MACD positivo"
            )



        if self.snapshot.get(
            "momentum_score",
            0
        ) >= 60:


            score += 7


        return score




    def short_momentum(self,reasons):


        score = 0


        if self.snapshot.get(
            "rsi",
            50
        ) < 50:


            score += 7



        if self.snapshot.get(
            "macd_hist",
            0
        ) < 0:


            score += 6



        if self.snapshot.get(
            "momentum_score",
            0
        ) < 40:


            score += 7


        return score





    # ============================
    # STRUCTURE
    # ============================


    def long_structure(self,reasons):


        score = 0


        if self.snapshot.get(
            "higher_high",
            False
        ):

            score += 10


        if self.snapshot.get(
            "higher_low",
            False
        ):

            score += 10



        if self.snapshot.get(
            "bos",
            False
        ):

            score += 5

            reasons.append(
                "BOS bullish"
            )


        return score




    def short_structure(self,reasons):


        score = 0


        if self.snapshot.get(
            "lower_high",
            False
        ):

            score += 10


        if self.snapshot.get(
            "lower_low",
            False
        ):

            score += 10



        if self.snapshot.get(
            "choch",
            False
        ):

            score += 5


        return score




    # ============================
    # REGIME
    # ============================


    def long_regime(self,reasons):


        if self.snapshot.get(
            "market_regime"
        ) in [
            "TREND_UP",
            "BREAKOUT"
        ]:

            reasons.append(
                "Regime favorável LONG"
            )

            return 15


        return 0




    def short_regime(self,reasons):


        if self.snapshot.get(
            "market_regime"
        ) == "TREND_DOWN":


            reasons.append(
                "Regime favorável SHORT"
            )

            return 15


        return 0




    # ============================
    # VOLUME
    # ============================


    def long_volume(self,reasons):


        score = 0


        if self.snapshot.get(
            "buy_pressure",
            0
        ) > 0.6:

            score += 8


        if self.snapshot.get(
            "institutional_score",
            0
        ) >= 60:

            score += 7


        return score




    def short_volume(self,reasons):


        score = 0


        if self.snapshot.get(
            "buy_pressure",
            0
        ) < 0.4:

            score += 8


        if self.snapshot.get(
            "institutional_score",
            0
        ) < 40:

            score += 7


        return score




    # ============================
    # PENALIDADES
    # ============================


    def calculate_penalties(
        self,
        warnings
    ):


        penalty = 0


        if self.snapshot.get(
            "market_regime"
        ) == "RANGE":


            penalty += 10

            warnings.append(
                "Mercado lateral"
            )



        if self.snapshot.get(
            "volume_ratio",
            1
        ) < 0.5:


            penalty += 10

            warnings.append(
                "Volume abaixo da média"
            )



        return penalty





    def calculate_confidence(
        self,
        score
    ):


        if score >= 85:

            return 80


        if score >= 70:

            return 70


        if score >= 55:

            return 60


        return 40





    def classify(
        self,
        score
    ):


        if score >= 85:

            return "STRONG_SIGNAL"


        if score >= 70:

            return "SIGNAL"


        if score >= 55:

            return "WAIT"


        return "NO_TRADE"





    def risk_level(
        self,
        score
    ):


        if score >= 80:

            return "LOW"


        if score >= 60:

            return "MEDIUM"


        return "HIGH"