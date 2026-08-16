import sqlite3
from datetime import datetime
import os


class MarketRepository:
    """
    Responsável por persistir dados de mercado
    e indicadores no banco analítico.
    """


    def __init__(self):

        self.database_path = (
            "data/market.db"
        )

        self.create_database()



    def connect(self):

        return sqlite3.connect(
            self.database_path
        )



    def create_database(self):

        os.makedirs(
            "data",
            exist_ok=True
        )


        connection = self.connect()

        cursor = connection.cursor()


        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS indicator_snapshots (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                symbol TEXT NOT NULL,

                timeframe TEXT NOT NULL,

                timestamp TEXT NOT NULL,


                close REAL,


                rsi REAL,

                stoch_rsi REAL,

                roc REAL,


                ema_fast REAL,

                ema_slow REAL,

                macd REAL,

                adx REAL,


                atr REAL,

                bb_high REAL,

                bb_low REAL,


                volume REAL,

                relative_volume REAL,

                volume_spike INTEGER,


                vwap REAL,

                mfi REAL,

                ad_line REAL,


                created_at TEXT

            )
            """
        )


        connection.commit()

        connection.close()



    def save_snapshot(
            self,
            data: dict
    ):


        connection = self.connect()

        cursor = connection.cursor()


        cursor.execute(
            """

            INSERT INTO indicator_snapshots (

                symbol,
                timeframe,
                timestamp,

                close,

                rsi,
                stoch_rsi,
                roc,

                ema_fast,
                ema_slow,
                macd,
                adx,

                atr,
                bb_high,
                bb_low,

                volume,
                relative_volume,
                volume_spike,

                vwap,
                mfi,
                ad_line,

                created_at

            )

            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)

            """,

            (

                data.get("symbol"),

                data.get("timeframe"),

                data.get("timestamp"),


                data.get("close"),


                data.get("rsi"),

                data.get("stoch_rsi"),

                data.get("roc"),


                data.get("ema_fast"),

                data.get("ema_slow"),

                data.get("macd"),

                data.get("adx"),


                data.get("atr"),

                data.get("bb_high"),

                data.get("bb_low"),


                data.get("volume"),

                data.get("relative_volume"),

                data.get("volume_spike"),


                data.get("vwap"),

                data.get("mfi"),

                data.get("ad_line"),


                datetime.utcnow().isoformat()

            )

        )


        connection.commit()

        connection.close()