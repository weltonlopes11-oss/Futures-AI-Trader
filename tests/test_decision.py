from decision.engine import AIDecisionEngine



market_analysis = {

    "market_regime": "TREND_UP"

}



signal_result = {


    "direction": "LONG",


    "long_score": 90,


    "short_score": 25,


    "confidence": 80,


    "risk": "LOW"


}



engine = AIDecisionEngine(

    market_analysis,

    signal_result

)



result = engine.decide()



print("\n================ AI DECISION RESULT ================\n")

print(result)