from trade.setup import TradeSetupEngine



print(
    "\n================ TESTE F.4.2 ================\n"
)



# =====================================================
# TESTE 1
# TRADE RUIM
# EXPECTATIVA: REJEITADO
# =====================================================


snapshot_bad = {


    "close": 64170,


    "atr_14": 12,


    "nearest_support": 64090,


    "nearest_resistance": 64218,


    "swing_low": 64080,


    "swing_high": 64300

}



decision_long = {


    "direction": "LONG"

}




engine_bad = TradeSetupEngine(

    snapshot_bad,

    decision_long

)



result_bad = engine_bad.generate()



print(
    "\n=========== TRADE RUIM ===========\n"
)


print(result_bad)



assert result_bad["setup"] is False


assert result_bad["status"] == "REJECTED"


assert result_bad["risk_reward"] < 1.5



print(
    "\n✓ Trade ruim rejeitado corretamente\n"
)





# =====================================================
# TESTE 2
# TRADE BOM
# EXPECTATIVA: APROVADO
# =====================================================


snapshot_good = {


    "close": 64170,


    "atr_14": 12,


    "nearest_support": 64080,


    "nearest_resistance": 64450,


    "swing_low": 64080,


    "swing_high": 64500

}




engine_good = TradeSetupEngine(

    snapshot_good,

    decision_long

)



result_good = engine_good.generate()



print(
    "\n=========== TRADE BOM ===========\n"
)


print(result_good)



assert result_good["setup"] is True


assert result_good["status"] == "APPROVED"


assert result_good["risk_reward"] >= 1.5


assert result_good["setup_quality"] in [

    "A",

    "B"

]



print(
    "\n✓ Trade bom aprovado corretamente\n"
)





# =====================================================
# TESTE 3
# VALIDAR SHORT
# =====================================================


snapshot_short = {


    "close": 64170,


    "atr_14": 12,


    "nearest_support": 63850,


    "nearest_resistance": 64250,


    "swing_low": 63800,


    "swing_high": 64280

}



decision_short = {


    "direction": "SHORT"

}




engine_short = TradeSetupEngine(

    snapshot_short,

    decision_short

)



result_short = engine_short.generate()



print(
    "\n=========== TRADE SHORT ===========\n"
)


print(result_short)



assert result_short["direction"] == "SHORT"


assert "risk_reward" in result_short



print(
    "\n✓ Direção SHORT validada\n"
)





print(
    "\n================ F.4.2 FINALIZADO COM SUCESSO ================\n"
)