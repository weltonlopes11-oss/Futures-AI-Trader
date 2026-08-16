from data.database import Database



db = Database()



print("\n================ TABELAS ================\n")


cursor = db.connection.cursor()


cursor.execute(

    """

    SELECT name

    FROM sqlite_master

    WHERE type='table'

    ORDER BY name

    """

)


tables = cursor.fetchall()


for table in tables:

    print(table[0])





print("\n================ MARKET ANALYSIS ================\n")


cursor.execute(

    """

    SELECT *

    FROM market_analysis_history

    ORDER BY id DESC

    LIMIT 5

    """

)


analysis = cursor.fetchall()


for row in analysis:

    print(row)





print("\n================ SIGNAL HISTORY ================\n")


cursor.execute(

    """

    SELECT *

    FROM signal_history

    ORDER BY id DESC

    LIMIT 5

    """

)


signals = cursor.fetchall()


for row in signals:

    print(row)





print("\n================ INDICATOR SNAPSHOTS ================\n")


cursor.execute(

    """

    SELECT COUNT(*)

    FROM indicator_snapshots

    """

)


count = cursor.fetchone()


print(

    "Snapshots registrados:",

    count[0]

)



db.close()