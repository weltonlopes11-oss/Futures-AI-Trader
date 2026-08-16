import pandas as pd

from indicators.engine import IndicatorsEngine
from market_data.indicator_storage import IndicatorStorage



data = {

    "open": [
        100,
        101,
        102,
        103,
        104
    ],

    "high": [
        102,
        103,
        104,
        105,
        106
    ],

    "low": [
        99,
        100,
        101,
        102,
        103
    ],

    "close": [
        101,
        102,
        103,
        104,
        105
    ],

    "volume": [
        1000,
        1200,
        1500,
        2000,
        3000
    ]

}



df = pd.DataFrame(data)



engine = IndicatorsEngine(df)


result = engine.calculate()



storage = IndicatorStorage()


storage.save(
    symbol="BTCUSDT",
    timeframe="1m",
    dataframe=result
)



print(
    "Snapshot salvo com sucesso"
)