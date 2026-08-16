from core.logging import configure_logging


def setup_logger():
    """
    Camada de compatibilidade para imports legados.

    Reutiliza integralmente a configuração oficial de logging
    definida em core.logging.
    """

    return configure_logging()
