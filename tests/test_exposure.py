from risk.exposure import ExposureController



print(
    "\n================ TESTE F.5.2 =================\n"
)



controller = ExposureController(

    capital=10000

)




# ==========================================
# TESTE 1
# APROVADO
# ==========================================


result_ok = controller.evaluate(

    position_size_value=1000,

    current_exposure=500,

    daily_loss=50,

    open_positions=1

)



print(
    "\n=========== EXPOSURE OK ===========\n"
)


print(result_ok)



assert result_ok["approved"] is True



print(
    "\n✓ Exposição aprovada\n"
)






# ==========================================
# TESTE 2
# EXPOSIÇÃO EXCEDIDA
# ==========================================


result_exposure = controller.evaluate(

    position_size_value=2500,

    current_exposure=1000,

    daily_loss=50,

    open_positions=1

)



print(
    "\n=========== EXPOSURE LIMIT ===========\n"
)


print(result_exposure)



assert result_exposure["approved"] is False



print(
    "\n✓ Exposição bloqueada\n"
)






# ==========================================
# TESTE 3
# DAILY LOSS
# ==========================================


result_loss = controller.evaluate(

    position_size_value=500,

    current_exposure=500,

    daily_loss=400,

    open_positions=1

)



print(
    "\n=========== DAILY LOSS ===========\n"
)


print(result_loss)



assert result_loss["approved"] is False



print(
    "\n✓ Perda diária bloqueada\n"
)






# ==========================================
# TESTE 4
# COOLDOWN
# ==========================================


result_cooldown = controller.evaluate(

    position_size_value=500,

    current_exposure=500,

    daily_loss=50,

    open_positions=1,

    cooldown_active=True

)



print(
    "\n=========== COOLDOWN ===========\n"
)


print(result_cooldown)



assert result_cooldown["approved"] is False



print(
    "\n✓ Cooldown validado\n"
)




print(
    "\n================ F.5.2 FINALIZADO =================\n"
)