from market_data.collector import MarketDataCollector

from indicators.engine import IndicatorsEngine

from features.manager import SnapshotManager

import pandas as pd

import json



# ==========================================
# INICIALIZAÇÃO
# ==========================================


collector = MarketDataCollector()



# ==========================================
# COLETA PREÇO ATUAL
# ==========================================


current_price = collector.collect_price()



print("\nPreço atual BTCUSDT:")
print(current_price)



# ==========================================
# COLETA CANDLES
# ==========================================


candles = collector.exchange.get_klines(

    "BTCUSDT",

    "1m",

    200

)



print("\nCandles recebidos:")

print(len(candles))



# ==========================================
# DATAFRAME
# ==========================================


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



print("\nDataFrame pronto")

print(df.tail())



# ==========================================
# ENGINE
# ==========================================


engine = IndicatorsEngine(df)


result = engine.calculate()



print(

    "\n================ INDICADORES OK ================\n"

)


print(result.tail())



# ==========================================
# VALIDA COLUNAS IMPORTANTES
# ==========================================


required = [

    "adx",

    "trend_strength",

    "market_structure",

    "bos",

    "choch",

    "market_regime"

]



print(

    "\n================ VALIDAÇÃO ================\n"

)



for item in required:


    if item in result.columns:

        print(
            f"[OK] {item}"
        )


    else:

        print(
            f"[ERRO] {item} ausente"
        )



# ==========================================
# SNAPSHOT IA
# ==========================================


manager = SnapshotManager()


snapshot = manager.save(

    result

)



print(

    "\n================ SNAPSHOT IA ================\n"

)



print(

    json.dumps(

        snapshot,

        indent=4,

        ensure_ascii=False

    )

)



# ==========================================
# MARKET DIAGNOSTIC
# ==========================================


print(

    "\n================ MARKET ANALYSIS ================\n"

)



print(

    f"""

Preço:

{snapshot['close']}



Tendência:

{snapshot['market_context']['trend']}



Força:

{snapshot['market_context']['strength']}



Estrutura:

{snapshot['market_context']['structure']}



Momentum:

{snapshot['market_context']['momentum']}



BOS:

{snapshot['bos']}



CHOCH:

{snapshot['choch']}



Market Regime:

{snapshot.get('market_regime','N/A')}

"""

)