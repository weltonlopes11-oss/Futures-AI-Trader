from data.database import Database


database = Database()

connection = database.connect()

cursor = connection.cursor()


cursor.execute(
    """
    SELECT *
    FROM market_prices
    ORDER BY id DESC
    LIMIT 10
    """
)


rows = cursor.fetchall()


print("\nÚltimos registros:\n")


for row in rows:

    print(row)


cursor.close()

connection.close()