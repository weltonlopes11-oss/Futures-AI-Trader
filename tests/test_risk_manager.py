from risk.manager import RiskManager



print(
    "\n================ TESTE F.5 RISK MANAGER ================\n"
)





# =====================================================
# TESTE 1
# RISCO APROVADO
# =====================================================


trade_good = {


    "setup": True,


    "direction": "LONG",


    "entry": 64170,


    "stop_loss": 64080,


    "take_profit": 64450,


    "risk_reward": 3.11


}




manager = RiskManager(

    capital=10000,

    risk_percent=1

)




result_good = manager.evaluate(

    trade_good

)



print(
    "\n=========== RISCO APROVADO ===========\n"
)


print(result_good)



assert result_good["approved"] is True


assert result_good["risk_status"] == "OK"


assert result_good["risk_amount"] == 100



print(
    "\n✓ Risco aprovado corretamente\n"
)







# =====================================================
# TESTE 2
# SETUP REJEITADO
# =====================================================


trade_rejected = {


    "setup": False,


    "direction": "LONG"

}





result_blocked = manager.evaluate(

    trade_rejected

)



print(
    "\n=========== SETUP BLOQUEADO ===========\n"
)


print(result_blocked)



assert result_blocked["approved"] is False


assert result_blocked["risk_status"] == "BLOCKED"



print(
    "\n✓ Setup rejeitado bloqueado corretamente\n"
)








# =====================================================
# TESTE 3
# RISCO ACIMA DO LIMITE
# =====================================================


manager_high_risk = RiskManager(

    capital=10000,

    risk_percent=5

)




result_high_risk = manager_high_risk.evaluate(

    trade_good

)



print(
    "\n=========== RISCO EXCEDIDO ===========\n"
)


print(result_high_risk)



assert result_high_risk["approved"] is False


assert result_high_risk["reason"] == "Risco excedido"



print(
    "\n✓ Limite de risco validado corretamente\n"
)






print(
    "\n================ F.5 FINALIZADO ================\n"
)