from logs.logger import setup_logger


logger = setup_logger()



class TradeSetupEngine:


    def __init__(
        self,
        snapshot,
        decision
    ):

        self.snapshot = snapshot
        self.decision = decision



    def generate(self):


        direction = self.decision.get(
            "direction"
        )


        if direction not in ["LONG", "SHORT"]:

            return {

                "setup": False,

                "status": "REJECTED",

                "reason": "Direção inválida"

            }



        entry = self.snapshot.get(
            "close"
        )


        atr = self.snapshot.get(
            "atr_14",
            0
        )


        support = self.snapshot.get(
            "nearest_support"
        )


        resistance = self.snapshot.get(
            "nearest_resistance"
        )


        swing_low = self.snapshot.get(
            "swing_low"
        )


        swing_high = self.snapshot.get(
            "swing_high"
        )



        if direction == "LONG":

            result = self.long_setup(

                entry,

                atr,

                support,

                resistance,

                swing_low

            )


        else:

            result = self.short_setup(

                entry,

                atr,

                support,

                resistance,

                swing_high

            )


        logger.info(
            f"Adaptive Trade Setup: {result}"
        )


        return result






    # ==================================================
    # LONG
    # ==================================================


    def long_setup(
        self,
        entry,
        atr,
        support,
        resistance,
        swing_low
    ):


        atr_stop = entry - (
            atr * 2
        )



        stop_loss = min(

            atr_stop,

            support,

            swing_low

        )



        take_profit = resistance



        return self.validate(

            "LONG",

            entry,

            stop_loss,

            take_profit

        )






    # ==================================================
    # SHORT
    # ==================================================


    def short_setup(
        self,
        entry,
        atr,
        support,
        resistance,
        swing_high
    ):


        atr_stop = entry + (

            atr * 2

        )



        stop_loss = max(

            atr_stop,

            resistance,

            swing_high

        )



        take_profit = support



        return self.validate(

            "SHORT",

            entry,

            stop_loss,

            take_profit

        )






    # ==================================================
    # VALIDATION
    # ==================================================


    def validate(
        self,
        direction,
        entry,
        stop_loss,
        take_profit
    ):


        risk = abs(

            entry - stop_loss

        )


        reward = abs(

            take_profit - entry

        )



        if risk == 0:


            return self.reject(

                entry,

                stop_loss,

                0,

                "Risco inválido"

            )



        risk_reward = reward / risk



        if risk_reward < 1.5:


            return self.reject(

                entry,

                stop_loss,

                risk_reward,

                "Estrutura não oferece R:R mínimo"

            )



        quality = "B"


        if risk_reward >= 2:

            quality = "A"



        return {


            "setup": True,

            "status": "APPROVED",

            "direction": direction,

            "entry": round(
                entry,
                2
            ),

            "stop_loss": round(
                stop_loss,
                2
            ),

            "take_profit": round(
                take_profit,
                2
            ),

            "risk_points": round(
                risk,
                2
            ),

            "reward_points": round(
                reward,
                2
            ),

            "risk_reward": round(
                risk_reward,
                2
            ),

            "setup_quality": quality

        }






    # ==================================================
    # REJECT
    # ==================================================


    def reject(
        self,
        entry,
        stop_loss,
        risk_reward,
        reason
    ):


        return {


            "setup": False,

            "status": "REJECTED",

            "entry": round(
                entry,
                2
            ),

            "stop_loss": round(
                stop_loss,
                2
            ),

            "risk_reward": round(
                risk_reward,
                2
            ),

            "setup_quality": "REJECTED",

            "reason": reason

        }