import os

from backtest.dataset_builder import HistoricalDatasetBuilder




print(

    "\n================ TESTE F.BT.1.5 =================\n"

)





builder = HistoricalDatasetBuilder(

    symbol="BTCUSDT"

)





# ===============================
# 1 HORA
# ===============================


df_1h = builder.build(

    interval="1h",

    limit=1500

)



print(

    "\n=========== BTC 1H ===========\n"

)


print(df_1h.tail())





# ===============================
# 4 HORAS
# ===============================


df_4h = builder.build(

    interval="4h",

    limit=1500

)



print(

    "\n=========== BTC 4H ===========\n"

)


print(df_4h.tail())





# ===============================
# DIÁRIO
# ===============================


df_1d = builder.build(

    interval="1d",

    limit=1500

)



print(

    "\n=========== BTC 1D ===========\n"

)


print(df_1d.tail())






# ===============================
# VALIDAÇÕES
# ===============================


assert len(df_1h) >0


assert len(df_4h) >0


assert len(df_1d) >0





assert os.path.exists(

    "data/historical/BTCUSDT_1h.csv"

)



assert os.path.exists(

    "data/historical/BTCUSDT_4h.csv"

)



assert os.path.exists(

    "data/historical/BTCUSDT_1d.csv"

)





print(

    "\n✓ Dataset 1H criado"

)



print(

    "✓ Dataset 4H criado"

)



print(

    "✓ Dataset 1D criado"

)





print(

    "\n================ F.BT.1.5 FINALIZADO =================\n"

)