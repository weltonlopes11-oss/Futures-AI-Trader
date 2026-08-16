from logs.logger import setup_logger


logger = setup_logger()



class ExposureController:


    def __init__(
        self,
        capital,
        max_exposure_percent=30,
        max_daily_loss_percent=3,
        max_positions=3,
        cooldown_minutes=30
    ):

        self.capital = capital

        self.max_exposure_percent = max_exposure_percent

        self.max_daily_loss_percent = max_daily_loss_percent

        self.max_positions = max_positions

        self.cooldown_minutes = cooldown_minutes





    def evaluate(
        self,
        position_size_value,
        current_exposure,
        daily_loss,
        open_positions,
        cooldown_active=False
    ):


        max_exposure = (

            self.capital *

            self.max_exposure_percent /

            100

        )


        max_daily_loss = (

            self.capital *

            self.max_daily_loss_percent /

            100

        )



        # ==========================
        # EXPOSURE
        # ==========================


        if (
            current_exposure + position_size_value
            >
            max_exposure
        ):

            return self.block(

                "Exposição máxima excedida"

            )




        # ==========================
        # DAILY LOSS
        # ==========================


        if daily_loss >= max_daily_loss:


            return self.block(

                "Limite diário de perda atingido"

            )




        # ==========================
        # POSITIONS
        # ==========================


        if open_positions >= self.max_positions:


            return self.block(

                "Máximo de posições abertas atingido"

            )





        # ==========================
        # COOLDOWN
        # ==========================


        if cooldown_active:


            return self.block(

                "Cooldown ativo após perda"

            )





        result = {


            "approved": True,

            "status": "OK",

            "available_exposure":

                round(

                    max_exposure -

                    current_exposure,

                    2

                )

        }



        logger.info(

            f"Exposure approved: {result}"

        )


        return result






    def block(
        self,
        reason
    ):


        result = {


            "approved": False,

            "status": "BLOCKED",

            "reason": reason

        }



        logger.warning(

            f"Exposure blocked: {reason}"

        )



        return result