from logs.logger import setup_logger


logger = setup_logger()



class RiskManager:


    def __init__(
        self,
        capital,
        risk_percent=1.0,
        max_risk_percent=2.0
    ):

        self.capital = capital

        self.risk_percent = risk_percent

        self.max_risk_percent = max_risk_percent





    def evaluate(
        self,
        trade_setup
    ):


        logger.info(
            "Risk Manager iniciado"
        )



        # =====================================
        # VALIDA SETUP
        # =====================================


        if not trade_setup.get(
            "setup",
            False
        ):


            return {


                "approved": False,


                "risk_status": "BLOCKED",


                "reason": "Setup rejeitado"


            }





        # =====================================
        # VALIDA RISCO
        # =====================================


        if self.risk_percent > self.max_risk_percent:


            return {


                "approved": False,


                "risk_status": "BLOCKED",


                "reason": "Risco excedido"


            }





        entry = trade_setup.get(
            "entry"
        )


        stop_loss = trade_setup.get(
            "stop_loss"
        )



        direction = trade_setup.get(
            "direction"
        )



        if entry is None or stop_loss is None:


            return {


                "approved": False,


                "risk_status": "BLOCKED",


                "reason": "Dados de entrada inválidos"


            }





        # =====================================
        # CALCULO FINANCEIRO
        # =====================================


        risk_amount = (

            self.capital *

            self.risk_percent /

            100

        )



        stop_distance = abs(

            entry - stop_loss

        )



        if stop_distance <= 0:


            return {


                "approved": False,


                "risk_status": "BLOCKED",


                "reason": "Stop inválido"


            }





        # =====================================
        # TAMANHO DA POSIÇÃO
        # =====================================


        position_size = (

            risk_amount /

            stop_distance

        )





        result = {


            "approved": True,


            "risk_status": "OK",


            "direction": direction,


            "capital": self.capital,


            "risk_percent": self.risk_percent,


            "risk_amount": round(

                risk_amount,

                2

            ),


            "entry": entry,


            "stop_loss": stop_loss,


            "stop_distance": round(

                stop_distance,

                2

            ),


            "position_size": round(

                position_size,

                6

            ),


            "max_loss": round(

                risk_amount,

                2

            )


        }



        logger.info(

            f"Risk Decision: {result}"

        )



        return result