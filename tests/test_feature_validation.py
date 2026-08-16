import pandas as pd

from backtest.feature_validator import FeatureValidator



print(
"\n================ TESTE F.BT.2 ================\n"
)



df = pd.read_csv(
"data/historical/BTCUSDT_multiframe.csv"
)



# Dados mínimos simulados
# até conectarmos os indicadores reais


df["market_regime"] = "RANGE"

df.loc[
    df.index % 3 == 0,
    "market_regime"
] = "TREND_UP"



df["rsi"] = 55

df["macd_hist"] = 1

df["momentum_score"] = 70



validator = FeatureValidator()



result = validator.run(
    df
)



print(
"\n=========== FEATURE VALIDATION ===========\n"
)


print(result)



assert "regime_analysis" in result

assert "momentum_analysis" in result



print(
"\n✓ Regime validado"
)


print(
"✓ Momentum validado"
)


print(
"\n================ F.BT.2 FINALIZADO ================\n"
)