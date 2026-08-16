from backtest.feature_injector import RealFeatureInjector



print(
"\n================ TESTE F.BT.2.1 ================\n"
)



injector = RealFeatureInjector()



df = injector.inject(
    "data/historical/BTCUSDT_multiframe.csv"
)



print(
"\n=========== REAL FEATURES ===========\n"
)


print(
df.tail()
)


print(
"\nColunas:"
)


print(
list(df.columns)
)



assert len(df) > 0



assert "timestamp" in df.columns



print(
"\n✓ Dataset enriquecido criado"
)


print(
"✓ Features reais aplicadas"
)


print(
"\n================ F.BT.2.1 FINALIZADO ================\n"
)