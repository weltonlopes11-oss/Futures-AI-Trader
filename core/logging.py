from pathlib import Path
from loguru import logger


_CONFIGURED = False


def configure_logging():
    """
    Configura o Loguru apenas uma vez para toda a aplicação.
    """

    global _CONFIGURED

    if _CONFIGURED:
        return logger

    project_root = Path(__file__).resolve().parent.parent

    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / "futures_ai_trader.log"

    logger.remove()

    logger.add(
        log_file,
        level="INFO",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        enqueue=False
    )

    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level:<8} | {message}"
    )

    _CONFIGURED = True

    logger.info("Sistema de logs inicializado")

    return logger