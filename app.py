from loguru import logger
from config.settings import settings


class Application:

    def start(self):

        logger.info(f"{settings.APP_NAME} iniciado")

        logger.info(f"Versão {settings.VERSION}")


application = Application()