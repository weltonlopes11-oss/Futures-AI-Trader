from backtest.regime_calibrator import RegimeCalibrator



print(
"\n================ TESTE F.BT.2.3 ================\n"
)



calibrator = RegimeCalibrator()



result = calibrator.run(
    "data/historical/BTCUSDT_features.csv"
)



print(
"\n=========== REGIME CALIBRATION REPORT ===========\n"
)


print(result)



assert "metrics" in result

assert "current_distribution" in result

assert "suggested_thresholds" in result



print(
"\n✓ Métricas calculadas"
)


print(
"✓ Distribuição analisada"
)


print(
"✓ Thresholds sugeridos"
)



print(
"\n================ F.BT.2.3 FINALIZADO ================\n"
)