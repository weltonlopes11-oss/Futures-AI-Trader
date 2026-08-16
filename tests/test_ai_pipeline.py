from market_data.collector import MarketDataCollector

from indicators.engine import IndicatorsEngine

from features.manager import SnapshotManager

from ai_engine.engine import AIEngine


import pandas as pd




collector = MarketDataCollector()



collector.collect_price()



candles = collector.exchange.get_klines(

    "BTCUSDT",

    "1m",

    200

)



columns = [

    "timestamp",

    "open",

    "high",

    "low",

    "close",

    "volume",

    "close_time",

    "quote_volume",

    "trades",

    "taker_buy_base",

    "taker_buy_quote",

    "ignore"

]



df = pd.DataFrame(

    candles,

    columns=columns

)



numeric_columns = [

    "open",

    "high",

    "low",

    "close",

    "volume",

    "quote_volume",

    "taker_buy_base",

    "taker_buy_quote"

]



for column in numeric_columns:


    df[column] = pd.to_numeric(

        df[column]

    )



# Indicadores

engine = IndicatorsEngine(

    df

)



result = engine.calculate()



# Snapshot

manager = SnapshotManager()



snapshot = manager.save(

    result

)



# IA

ai = AIEngine()



analysis = ai.analyze_market(

    snapshot

)



print("\n================ AI RESULT ================\n")


print(

    analysis

)