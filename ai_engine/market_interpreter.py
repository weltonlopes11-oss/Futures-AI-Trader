from logs.logger import setup_logger


logger = setup_logger()



class MarketInterpreter:


    def __init__(self):

        logger.info(
            "AI Market Interpreter iniciado"
        )



    def analyze(self, snapshot):


        trend = snapshot.get(
            "trend",
            0
        )


        adx = snapshot.get(
            "adx",
            0
        )


        momentum = snapshot.get(
            "momentum_score",
            0
        )


        bos = snapshot.get(
            "bos",
            False
        )


        choch = snapshot.get(
            "choch",
            False
        )


        regime = snapshot.get(
            "market_regime",
            "UNKNOWN"
        )


        structure = snapshot.get(
            "market_structure",
            "UNKNOWN"
        )



        market_state = "UNCERTAIN"

        bias = "WAIT"

        confidence = 30


        reasons = []



        # ==================================
        # TENDÊNCIA DE ALTA
        # ==================================

        if (

            trend > 0

            and adx >= 25

            and bos is True

        ):


            market_state = "TRENDING_UP"

            bias = "BUY"

            confidence = 75


            reasons.append(
                "Tendência de alta com rompimento estrutural confirmado"
            )



        # ==================================
        # TENDÊNCIA DE BAIXA
        # ==================================

        elif (

            trend < 0

            and adx >= 25

            and choch is True

        ):


            market_state = "TRENDING_DOWN"

            bias = "SELL"

            confidence = 75


            reasons.append(
                "Estrutura de baixa confirmada"
            )



        # ==================================
        # RANGE
        # ==================================

        elif regime == "RANGE":


            market_state = "RANGE"

            bias = "WAIT"

            confidence = 60


            reasons.append(
                "Mercado lateral sem confirmação estrutural"
            )



        # ==================================
        # REVERSÃO
        # ==================================

        elif choch is True:


            market_state = "REVERSAL_ZONE"

            bias = "WAIT"

            confidence = 55


            reasons.append(
                "Possível mudança de estrutura detectada"
            )



        # ==================================
        # AJUSTE POR MOMENTUM
        # ==================================

        if momentum >= 70:

            confidence += 10


            reasons.append(
                "Momentum positivo"
            )


        elif momentum <= 30:

            confidence -= 10


            reasons.append(
                "Momentum fraco"
            )



        confidence = max(
            0,
            min(
                confidence,
                100
            )
        )



        result = {


            "market_state":

                market_state,


            "bias":

                bias,


            "confidence":

                confidence,


            "analysis":

            {


                "trend":

                    trend,


                "adx":

                    adx,


                "momentum_score":

                    momentum,


                "market_regime":

                    regime,


                "market_structure":

                    structure,


                "bos":

                    bos,


                "choch":

                    choch

            },


            "reasons":

                reasons

        }



        logger.info(
            f"AI Market Interpretation: {result}"
        )



        return result