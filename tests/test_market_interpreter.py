from ai_engine.market_interpreter import MarketInterpreter


snapshot = {


    "trend": 1,

    "adx": 34.6,

    "momentum_score": 15,

    "bos": False,

    "choch": False,

    "market_regime": "RANGE",

    "market_structure": "NEUTRAL_STRUCTURE"

}



interpreter = MarketInterpreter()



result = interpreter.analyze(

    snapshot

)



print("\n================ AI MARKET INTERPRETER ================\n")


print(

    f"""
Estado do mercado:

{result['market_state']}


Viés:

{result['bias']}


Confiança:

{result['confidence']}


Motivos:

"""

)



for reason in result["reasons"]:

    print(
        "-",
        reason
    )



print("\n================ DETALHES ================\n")


print(result)