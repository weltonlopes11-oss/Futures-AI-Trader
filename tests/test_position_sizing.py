from risk.position_sizing import DynamicPositionSizer



print(
    "\n================ TESTE F.5.1 =================\n"
)





trade_setup = {


    "setup": True,


    "direction": "LONG",


    "entry": 64170,


    "stop_loss": 64080,


    "setup_quality": "A"


}






# ==================================================
# CENÁRIO 1
# SETUP A + ALTA CONFIANÇA
# ==================================================


sizer = DynamicPositionSizer(

    capital=10000

)



result_a = sizer.calculate(

    trade_setup,

    confidence=90,

    volatility_percent=0.015

)



print(
    "\n=========== SETUP A ===========\n"
)


print(result_a)



assert result_a["approved"] is True


assert result_a["risk_percent"] > 1



print(
    "\n✓ Setup A aumentou risco corretamente\n"
)






# ==================================================
# CENÁRIO 2
# BAIXA CONFIANÇA
# ==================================================


result_low = sizer.calculate(

    trade_setup,

    confidence=50,

    volatility_percent=0.06

)



print(
    "\n=========== BAIXA CONFIANÇA ===========\n"
)


print(result_low)



assert result_low["approved"] is True


assert result_low["risk_percent"] < result_a["risk_percent"]



print(
    "\n✓ Risco reduzido corretamente\n"
)






# ==================================================
# CENÁRIO 3
# VOLATILIDADE EXTREMA
# ==================================================


result_vol = sizer.calculate(

    trade_setup,

    confidence=90,

    volatility_percent=0.10

)



print(
    "\n=========== ALTA VOLATILIDADE ===========\n"
)


print(result_vol)



assert result_vol["risk_percent"] <= 1.25



print(
    "\n✓ Controle de volatilidade validado\n"
)






print(
    "\n================ F.5.1 FINALIZADO =================\n"
)