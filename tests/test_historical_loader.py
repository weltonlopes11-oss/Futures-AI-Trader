from backtest.historical_loader import HistoricalDataLoader





print(

    "\n================ TESTE F.BT.1 =================\n"

)





loader = HistoricalDataLoader(

    symbol="BTCUSDT",

    interval="1m"

)





df = loader.load(

    limit=500

)





print(

    "\n=========== HISTÓRICO RECEBIDO ===========\n"

)



print(df.head())



print("\nQuantidade candles:")

print(len(df))



print("\nÚltimos candles:")

print(df.tail())





# ==========================
# VALIDAÇÕES
# ==========================


assert len(df) == 500


assert "close" in df.columns


assert "volume" in df.columns


assert df["close"].dtype != "object"





print(

    "\n✓ Quantidade correta\n"

)



print(

    "✓ Colunas validadas\n"

)



print(

    "✓ Conversão numérica validada\n"

)





print(

    "\n================ F.BT.1 FINALIZADO =================\n"

)