from signals.scoring import SignalScoringEngine


snapshot = {

    "trend": 1,

    "adx": 32,

    "rsi": 58,

    "macd_hist": 3.2,

    "momentum_score": 75,

    "bos": True,

    "higher_high": True,

    "higher_low": True,

    "market_regime": "TREND_UP",

    "buy_pressure": 0.75,

    "institutional_score": 70

}



analysis = {

    "market_state": "TREND_UP"

}



engine = SignalScoringEngine(

    snapshot,

    analysis

)



result = engine.calculate()



print("\n================ SIGNAL RESULT ================\n")

print(result)