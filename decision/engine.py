from logs.logger import setup_logger


logger = setup_logger()



class AIDecisionEngine:


    def __init__(
        self,
        market_analysis,
        signal_result
    ):

        self.market_analysis = market_analysis
        self.signal_result = signal_result



    def decide(self):


        direction = self.signal_result.get(
            "direction",
            "NEUTRAL"
        )


        confidence = self.signal_result.get(
            "confidence",
            0
        )


        risk = self.signal_result.get(
            "risk",
            "HIGH"
        )


        regime = self.market_analysis.get(
            "market_regime",
            "UNKNOWN"
        )


        long_score = self.signal_result.get(
            "long_score",
            0
        )


        short_score = self.signal_result.get(
            "short_score",
            0
        )



        action = "WAIT"

        opportunity = False


        grade = "C"



        reasons = []



        # ==========================
        # LONG
        # ==========================


        if (

            direction == "LONG"

            and confidence >= 70

            and risk != "HIGH"

            and regime not in [
                "RANGE",
                "UNKNOWN"
            ]

            and long_score > short_score

        ):


            opportunity = True

            action = "PREPARE_LONG"

            reasons.append(
                "Condições LONG confirmadas"
            )



        # ==========================
        # SHORT
        # ==========================


        elif (

            direction == "SHORT"

            and confidence >= 70

            and risk != "HIGH"

            and regime not in [
                "RANGE",
                "UNKNOWN"
            ]

            and short_score > long_score

        ):


            opportunity = True

            action = "PREPARE_SHORT"

            reasons.append(
                "Condições SHORT confirmadas"
            )



        else:


            reasons.append(
                "Sem confirmação suficiente"
            )



        # ==========================
        # QUALIDADE
        # ==========================


        final_score = max(
            long_score,
            short_score
        )


        if final_score >= 85:

            grade = "A"


        elif final_score >= 70:

            grade = "B"



        result = {


            "opportunity": opportunity,


            "action": action,


            "direction": direction,


            "grade": grade,


            "confidence": confidence,


            "risk": risk,


            "market_regime": regime,


            "scores": {

                "long": long_score,

                "short": short_score

            },


            "reasons": reasons


        }



        logger.info(
            f"AI Decision: {result}"
        )


        return result