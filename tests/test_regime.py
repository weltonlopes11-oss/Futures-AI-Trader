from market_data.collector import MarketDataCollector

from indicators.engine import IndicatorsEngine

from features.manager import FeatureManager

import pandas as pd



collector = MarketDataCollector()



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



numeric = [

    "open",

    "high",

    "low",

    "close",

    "volume",

    "quote_volume",

    "taker_buy_base",

    "taker_buy_quote"

]


for col in numeric:

    df[col] = pd.to_numeric(
        df[col]
    )



engine = IndicatorsEngine(df)



result = engine.calculate()



features = FeatureManager(

    result

)



final = features.calculate()



print("\n================ REGIME RESULT ================\n")


print(

    final[

        [

            "close",

            "adx",

            "volume_ratio",

            "market_regime",

            "regime_score"

        ]

    ].tail()

)


print("\nÚltimo regime:")


print(

    final.iloc[-1]["market_regime"]

)


print("\nScore:")


print(

    final.iloc[-1]["regime_score"]

)