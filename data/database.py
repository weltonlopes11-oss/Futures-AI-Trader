import sqlite3

from logs.logger import setup_logger


logger = setup_logger()



class Database:


    def __init__(self):


        self.db_path = "market.db"


        self.connection = sqlite3.connect(
            self.db_path
        )


        logger.info(
            "Banco conectado"
        )



    # =====================================
    # CRIAÇÃO DAS TABELAS
    # =====================================


    def create_tables(self):


        cursor = self.connection.cursor()



        # ==========================
        # PREÇOS
        # ==========================


        cursor.execute(

            """

            CREATE TABLE IF NOT EXISTS prices

            (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                symbol TEXT NOT NULL,

                price REAL NOT NULL,

                timestamp TEXT NOT NULL

            )

            """

        )



        logger.info(
            "Tabela prices criada/verificada"
        )



        # ==========================
        # INDICADORES
        # ==========================


        cursor.execute(

            """

            CREATE TABLE IF NOT EXISTS indicator_snapshots

            (

                id INTEGER PRIMARY KEY AUTOINCREMENT,


                symbol TEXT NOT NULL,


                timestamp TEXT NOT NULL,


                close REAL,


                ema_20 REAL,

                ema_50 REAL,

                ema_200 REAL,


                trend REAL,


                atr_14 REAL,

                volatility_percent REAL,


                volume_sma_20 REAL,

                volume_ratio REAL,

                volume_spike INTEGER,


                rsi REAL,


                macd REAL,

                macd_signal REAL,

                macd_hist REAL,


                stoch_rsi REAL,


                momentum_score REAL,


                institutional_candle INTEGER,


                buy_pressure REAL,


                institutional_score REAL,


                created_at TEXT DEFAULT CURRENT_TIMESTAMP


            )

            """

        )


        logger.info(
            "Tabela indicator_snapshots criada/verificada"
        )



        # ==========================
        # MARKET ANALYSIS HISTORY
        # ==========================


        cursor.execute(

            """

            CREATE TABLE IF NOT EXISTS market_analysis_history

            (

                id INTEGER PRIMARY KEY AUTOINCREMENT,


                symbol TEXT NOT NULL,


                timestamp TEXT NOT NULL,


                market_state TEXT,


                bias TEXT,


                confidence REAL,


                reason TEXT,


                created_at TEXT DEFAULT CURRENT_TIMESTAMP

            )

            """

        )


        logger.info(
            "Tabela market_analysis_history criada/verificada"
        )



        # ==========================
        # SIGNAL HISTORY
        # ==========================


        cursor.execute(

            """

            CREATE TABLE IF NOT EXISTS signal_history

            (

                id INTEGER PRIMARY KEY AUTOINCREMENT,


                symbol TEXT NOT NULL,


                timestamp TEXT NOT NULL,


                signal TEXT,


                long_score REAL,


                short_score REAL,


                confidence REAL,


                created_at TEXT DEFAULT CURRENT_TIMESTAMP

            )

            """

        )


        logger.info(
            "Tabela signal_history criada/verificada"
        )



        # ==========================
        # TRADE JOURNAL
        # ==========================


        cursor.execute(

            """

            CREATE TABLE IF NOT EXISTS trade_journal

            (

                id INTEGER PRIMARY KEY AUTOINCREMENT,


                symbol TEXT NOT NULL,


                entry_price REAL,


                exit_price REAL,


                side TEXT,


                result REAL,


                entry_reason TEXT,


                exit_reason TEXT,


                created_at TEXT DEFAULT CURRENT_TIMESTAMP

            )

            """

        )


        logger.info(
            "Tabela trade_journal criada/verificada"
        )



        self.connection.commit()





    # =====================================
    # SALVAR PREÇO
    # =====================================


    def save_price(
        self,
        symbol,
        price
    ):


        cursor = self.connection.cursor()



        cursor.execute(

            """

            INSERT INTO prices

            (

                symbol,

                price,

                timestamp

            )

            VALUES

            (

                ?,

                ?,

                datetime('now')

            )

            """,

            (
                symbol,
                price
            )

        )


        self.connection.commit()



        logger.info(
            "Preço salvo no banco"
        )





    # =====================================
    # BUSCAR PREÇOS
    # =====================================


    def get_prices(
        self,
        symbol="BTCUSDT",
        limit=200
    ):


        cursor = self.connection.cursor()



        cursor.execute(

            """

            SELECT

                price,

                timestamp


            FROM prices


            WHERE symbol = ?


            ORDER BY id DESC


            LIMIT ?

            """,

            (
                symbol,
                limit
            )

        )


        return cursor.fetchall()





    # =====================================
    # SALVAR SNAPSHOT INDICADORES
    # =====================================


    def save_indicator_snapshot(
        self,
        snapshot
    ):


        cursor = self.connection.cursor()



        cursor.execute(

            """

            INSERT INTO indicator_snapshots

            (

                symbol,

                timestamp,

                close,

                ema_20,

                ema_50,

                ema_200,

                trend,

                atr_14,

                volatility_percent,

                volume_sma_20,

                volume_ratio,

                volume_spike,

                rsi,

                macd,

                macd_signal,

                macd_hist,

                stoch_rsi,

                momentum_score,

                institutional_candle,

                buy_pressure,

                institutional_score

            )


            VALUES

            (

                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?

            )


            """,

            (

                snapshot["symbol"],

                snapshot["timestamp"],

                snapshot.get("close"),

                snapshot.get("ema_20"),

                snapshot.get("ema_50"),

                snapshot.get("ema_200"),

                snapshot.get("trend"),

                snapshot.get("atr_14"),

                snapshot.get("volatility_percent"),

                snapshot.get("volume_sma_20"),

                snapshot.get("volume_ratio"),

                snapshot.get("volume_spike"),

                snapshot.get("rsi"),

                snapshot.get("macd"),

                snapshot.get("macd_signal"),

                snapshot.get("macd_hist"),

                snapshot.get("stoch_rsi"),

                snapshot.get("momentum_score"),

                snapshot.get("institutional_candle"),

                snapshot.get("buy_pressure"),

                snapshot.get("institutional_score")

            )

        )


        self.connection.commit()


        logger.info(
            "Indicator snapshot salvo com sucesso"
        )





    # =====================================
    # SALVAR ANÁLISE DA IA
    # =====================================


    def save_market_analysis(
        self,
        analysis,
        symbol="BTCUSDT"
    ):


        cursor = self.connection.cursor()



        cursor.execute(

            """

            INSERT INTO market_analysis_history

            (

                symbol,

                timestamp,

                market_state,

                bias,

                confidence,

                reason

            )


            VALUES

            (

                ?,

                datetime('now'),

                ?,

                ?,

                ?,

                ?

            )

            """,

            (

                symbol,

                analysis.get("market_state"),

                analysis.get("bias"),

                analysis.get("confidence"),

                str(
                    analysis.get("reasons")
                )

            )

        )


        self.connection.commit()



        logger.info(
            "Market analysis salvo"
        )





    # =====================================
    # SALVAR SINAL
    # =====================================


    def save_signal(
        self,
        signal_data,
        symbol="BTCUSDT"
    ):


        cursor = self.connection.cursor()



        cursor.execute(

            """

            INSERT INTO signal_history

            (

                symbol,

                timestamp,

                signal,

                long_score,

                short_score,

                confidence

            )


            VALUES

            (

                ?,

                datetime('now'),

                ?,

                ?,

                ?,

                ?

            )

            """,

            (

                symbol,

                signal_data.get("signal"),

                signal_data.get("long_score"),

                signal_data.get("short_score"),

                signal_data.get("confidence")

            )

        )


        self.connection.commit()



        logger.info(
            "Signal salvo"
        )





    # =====================================
    # SALVAR TRADE
    # =====================================


    def save_trade(
        self,
        trade
    ):


        cursor = self.connection.cursor()



        cursor.execute(

            """

            INSERT INTO trade_journal

            (

                symbol,

                entry_price,

                exit_price,

                side,

                result,

                entry_reason,

                exit_reason

            )


            VALUES

            (?,?,?,?,?,?,?)

            """,

            (

                trade.get("symbol"),

                trade.get("entry_price"),

                trade.get("exit_price"),

                trade.get("side"),

                trade.get("result"),

                trade.get("entry_reason"),

                trade.get("exit_reason")

            )

        )


        self.connection.commit()


        logger.info(
            "Trade registrado"
        )





    # =====================================
    # FECHAR BANCO
    # =====================================


    def close(self):


        self.connection.close()


        logger.info(
            "Banco fechado"
        )