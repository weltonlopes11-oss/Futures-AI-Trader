from logs.logger import setup_logger


logger = setup_logger()



class DynamicPositionSizer:


    def __init__(
        self,
        capital,
        base_risk_percent=1.0,
        max_risk_percent=2.0
    ):

        self.capital = capital

        self.base_risk_percent = base_risk_percent

        self.max_risk_percent = max_risk_percent





    def calculate(
        self,
        trade_setup,
        confidence,
        volatility_percent
    ):


        setup_quality = trade_setup.get(
            "setup_quality",
            "C"
        )


        risk_percent = self.adjust_risk(

            setup_quality,

            confidence,

            volatility_percent

        )



        risk_amount = (

            self.capital *

            risk_percent /

            100

        )



        entry = trade_setup["entry"]


        stop_loss = trade_setup["stop_loss"]



        stop_distance = abs(

            entry - stop_loss

        )



        if stop_distance <= 0:


            return {


                "approved": False,

                "reason": "Stop inválido"

            }



        position_size = (

            risk_amount /

            stop_distance

        )



        result = {


            "approved": True,


            "risk_percent": round(

                risk_percent,

                2

            ),


            "risk_amount": round(

                risk_amount,

                2

            ),


            "position_size": round(

                position_size,

                6

            ),


            "confidence": confidence,


            "setup_quality": setup_quality



        }



        logger.info(

            f"Dynamic Position Size: {result}"

        )



        return result






    def adjust_risk(
        self,
        setup_quality,
        confidence,
        volatility
    ):


        risk = self.base_risk_percent



        # ==========================
        # QUALIDADE DO SETUP
        # ==========================


        if setup_quality == "A":

            risk += 0.5


        elif setup_quality == "C":

            risk -= 0.5






        # ==========================
        # CONFIANÇA DA IA
        # ==========================


        if confidence >= 85:

            risk += 0.25


        elif confidence < 60:

            risk -= 0.25






        # ==========================
        # VOLATILIDADE
        # ==========================


        if volatility_percent := volatility:


            if volatility_percent > 0.05:

                risk -= 0.5


            elif volatility_percent < 0.02:

                risk += 0.25






        # limites


        risk = max(

            0.25,

            risk

        )


        risk = min(

            self.max_risk_percent,

            risk

        )



        return risk