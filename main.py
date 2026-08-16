from core.logging import configure_logging
from app import application


def main():

    configure_logging()

    application.start()


if __name__ == "__main__":
    main()