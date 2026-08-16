from features.regime import MarketRegimeDetector
import pandas as pd



print(
"\n================ TESTE F.BT.2.4 ================\n"
)



df = pd.read_csv(
    "data/historical/BTCUSDT_features.csv"
)



detector = MarketRegimeDetector(
    df
)


result = detector.calculate()



print(
"\n=========== REGIME V3 DISTRIBUTION ===========\n"
)


print(
result["market_regime"]
.value_counts()
)



print(
"\n=========== SCORE ===========\n"
)


print(
result[
[
"market_regime",
"regime_score"
]
]
.tail()
)



distribution = (
    result["market_regime"]
    .value_counts()
    .to_dict()
)



print(
"\nDistribuição:"
)

print(
distribution
)



assert (
    "market_regime"
    in result.columns
)


assert (
    len(result)
    >0
)



print(
"\n✓ Regime v3 calculado"
)


print(
"✓ Distribuição criada"
)


print(
"✓ Scores gerados"
)



print(
"\n================ F.BT.2.4 FINALIZADO ================\n"
)