from backtest.timeframe_merger import MultiTimeframeDataset



print(
    "\n================ TESTE F.BT.1.6 ================\n"
)



builder = MultiTimeframeDataset()



df = builder.merge(
    "BTCUSDT"
)



print(
    "\n=========== DATASET MULTI ===========\n"
)


print(df.tail())


print("\nColunas:\n")

print(
    list(df.columns)
)



assert len(df) > 0


assert "close_1h" in df.columns

assert "close_4h" in df.columns

assert "close_1d" in df.columns



assert df["timestamp"].is_monotonic_increasing



print(
    "\n✓ Dataset criado"
)


print(
    "✓ Timeframe 1H integrado"
)


print(
    "✓ Timeframe 4H integrado"
)


print(
    "✓ Timeframe 1D integrado"
)


print(
    "✓ Ordenação temporal validada"
)



print(
    "\n================ F.BT.1.6 FINALIZADO ================\n"
)