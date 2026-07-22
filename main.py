"""Point d'entrée principal de LotoFoot AI Analyst."""

from loguru import logger
from database.connection import init_db


def main():
    logger.info("=== LotoFoot AI Analyst ===")
    logger.info("Initialisation de la base de données...")
    init_db()
    logger.info("Base de données prête.")
    logger.info("Système initialisé avec succès.")


if __name__ == "__main__":
    main()
