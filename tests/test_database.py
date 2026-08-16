from data.database import Database


database = Database()


database.create_tables()


database.save_price(
    "BTCUSDT",
    118000.50
)


database.close()


print(
    "Banco criado e teste concluído"
)