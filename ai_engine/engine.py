from ai_engine.market_interpreter import MarketInterpreter

from logs.logger import setup_logger



logger = setup_logger()



class AIEngine:


    def __init__(self):


        self.market_interpreter = MarketInterpreter()


        logger.info(
            "AI Engine inicializado"
        )



    def analyze_market(self, snapshot):


        if not snapshot:


            raise ValueError(
                "Snapshot vazio para análise"
            )



        market_analysis = self.market_interpreter.analyze(

            snapshot

        )



        result = {


            "snapshot": snapshot,


            "market_analysis": market_analysis

        }



        logger.info(

            "Análise de mercado concluída"

        )



        return result