import sqlite3
import json

from logs.logger import setup_logger

logger = setup_logger()


class Database:

    def __init__(self):

        self.db_path = "market.db"

        self.connection = sqlite3.connect(self.db_path)

        logger.info("Banco conectado")

    def create_tables(self):

        cursor = self.connection.cursor()

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS prices(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            symbol TEXT NOT NULL,

            price REAL NOT NULL,

            timestamp TEXT NOT NULL

        )

        """)

        logger.info("Tabela prices criada/verificada")

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS indicator_snapshots(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            symbol TEXT NOT NULL,

            timestamp TEXT NOT NULL,

            close REAL,

            snapshot_json TEXT NOT NULL,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP

        )

        """)

        logger.info("Tabela indicator_snapshots criada/verificada")

        self.connection.commit()

    # =========================================

    def save_price(self, symbol, price):

        cursor = self.connection.cursor()

        cursor.execute("""

        INSERT INTO prices(

            symbol,

            price,

            timestamp

        )

        VALUES(

            ?,

            ?,

            datetime('now')

        )

        """, (symbol, price))

        self.connection.commit()

        logger.info("Preço salvo no banco")

    # =========================================

    def get_prices(self, symbol="BTCUSDT", limit=200):

        cursor = self.connection.cursor()

        cursor.execute("""

        SELECT

            price,

            timestamp

        FROM prices

        WHERE symbol=?

        ORDER BY id DESC

        LIMIT ?

        """, (symbol, limit))

        return cursor.fetchall()

    # =========================================

    def save_indicator_snapshot(self, snapshot):

        cursor = self.connection.cursor()

        symbol = snapshot["symbol"]

        timestamp = snapshot["timestamp"]

        close = snapshot["close"]

        snapshot_json = json.dumps(

            snapshot,

            ensure_ascii=False

        )

        cursor.execute("""

        INSERT INTO indicator_snapshots(

            symbol,

            timestamp,

            close,

            snapshot_json

        )

        VALUES(

            ?,

            ?,

            ?,

            ?

        )

        """, (

            symbol,

            timestamp,

            close,

            snapshot_json

        ))

        self.connection.commit()

        logger.info("Indicator snapshot salvo com sucesso")

    # =========================================

    def get_indicator_snapshots(self, limit=100):

        cursor = self.connection.cursor()

        cursor.execute("""

        SELECT

            id,

            symbol,

            timestamp,

            close,

            snapshot_json,

            created_at

        FROM indicator_snapshots

        ORDER BY id DESC

        LIMIT ?

        """, (limit,))

        rows = cursor.fetchall()

        snapshots = []

        for row in rows:

            snapshots.append({

                "id": row[0],

                "symbol": row[1],

                "timestamp": row[2],

                "close": row[3],

                "snapshot": json.loads(row[4]),

                "created_at": row[5]

            })

        return snapshots

    # =========================================

    def close(self):

        self.connection.close()

        logger.info("Banco fechado")