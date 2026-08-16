from backtest.real_feature_validator import RealFeatureValidator



print(
"\n================ TESTE F.BT.2.2 ================\n"
)



validator = RealFeatureValidator()



result = validator.run(
    "data/historical/BTCUSDT_features.csv"
)



print(
"\n=========== REAL FEATURE REPORT ===========\n"
)



print(result)



assert result["total_samples"] > 0


assert "regime_distribution" in result


assert "regime_analysis" in result



print(
"\n✓ Dataset real validado"
)


print(
"✓ Regimes analisados"
)


print(
"✓ Regime score analisado"
)



print(
"\n================ F.BT.2.2 FINALIZADO ================\n"
)