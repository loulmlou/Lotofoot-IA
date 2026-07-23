"""Point d'entrée principal de LotoFoot AI Analyst."""

import sys

from loguru import logger
from database.connection import init_db


def main():
    logger.info("=== LotoFoot AI Analyst ===")
    logger.info("Initialisation de la base de données...")
    init_db()
    logger.info("Base de données prête.")
    logger.info("Système initialisé avec succès.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "ui":
        import subprocess
        subprocess.run(["streamlit", "run", "frontend/app.py"])
    else:
        main()
