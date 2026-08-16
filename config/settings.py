import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    APP_NAME = "Futures AI Trader"
    VERSION = "0.1.0"

    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")


settings = Settings()