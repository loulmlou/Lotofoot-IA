"""Script orchestrateur de collecte des données.

Lance séquentiellement :
1. Téléchargement des CSV football-data.co.uk
2. Import en base de données SQLite
3. Scraping des grilles Loto Foot (FDJ)
4. Rapport de synthèse
"""

import sys
import os

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger
from sqlalchemy import func, select

from database.connection import init_db, SessionLocal
from database.models import Match, Equipe, Competition, Cote, GrilleLotoFoot
from collectors.football_data import collect_all as collect_football
from collectors.import_football import import_all as import_football
from collectors.lotofoot_scraper import scrape_all as scrape_lotofoot


def print_report(session) -> None:
    """Affiche un rapport de synthèse des données en base."""
    nb_matchs = session.execute(select(func.count(Match.id))).scalar() or 0
    nb_equipes = session.execute(select(func.count(Equipe.id))).scalar() or 0
    nb_competitions = session.execute(select(func.count(Competition.id))).scalar() or 0
    nb_cotes = session.execute(select(func.count(Cote.id))).scalar() or 0
    nb_grilles = session.execute(select(func.count(GrilleLotoFoot.id))).scalar() or 0

    # Période couverte
    date_min = session.execute(select(func.min(Match.date))).scalar()
    date_max = session.execute(select(func.max(Match.date))).scalar()

    print("\n" + "=" * 60)
    print("          RAPPORT DE SYNTHÈSE — COLLECTE")
    print("=" * 60)
    print(f"  Matchs en base         : {nb_matchs:>8}")
    print(f"  Équipes                : {nb_equipes:>8}")
    print(f"  Compétitions           : {nb_competitions:>8}")
    print(f"  Cotes                  : {nb_cotes:>8}")
    print(f"  Grilles Loto Foot      : {nb_grilles:>8}")
    if date_min and date_max:
        print(f"  Période                : {date_min} → {date_max}")
    print("=" * 60 + "\n")


def run(
    skip_football_download: bool = False,
    skip_football_import: bool = False,
    skip_lotofoot: bool = False,
):
    """Lance la collecte complète.

    Args:
        skip_football_download: Passer le téléchargement des CSV.
        skip_football_import: Passer l'import en base.
        skip_lotofoot: Passer le scraping Loto Foot.
    """
    logger.info("=== Démarrage de la collecte complète ===")

    # Initialiser la base
    init_db()

    # Étape 1 : Téléchargement CSV
    if not skip_football_download:
        logger.info("--- Étape 1 : Téléchargement des CSV football-data.co.uk ---")
        football_stats = collect_football()
        logger.info(f"Téléchargement : {football_stats}")
    else:
        logger.info("--- Étape 1 : Téléchargement ignoré (skip_football_download) ---")

    # Étape 2 : Import en base
    if not skip_football_import:
        logger.info("--- Étape 2 : Import en base de données ---")
        import_stats = import_football()
        logger.info(f"Import : {import_stats}")
    else:
        logger.info("--- Étape 2 : Import ignoré (skip_football_import) ---")

    # Étape 3 : Scraping Loto Foot
    if not skip_lotofoot:
        logger.info("--- Étape 3 : Scraping Loto Foot (FDJ) ---")
        lotofoot_stats = scrape_lotofoot()
        logger.info(f"Scraping LotoFoot : {lotofoot_stats}")
    else:
        logger.info("--- Étape 3 : Scraping Loto Foot ignoré (skip_lotofoot) ---")

    # Étape 4 : Rapport
    logger.info("--- Étape 4 : Rapport de synthèse ---")
    session = SessionLocal()
    try:
        print_report(session)
    finally:
        session.close()

    logger.info("=== Collecte terminée ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collecte des données LotoFoot")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Ne pas télécharger les CSV (utiliser ceux existants)"
    )
    parser.add_argument(
        "--skip-import", action="store_true",
        help="Ne pas importer les CSV en base"
    )
    parser.add_argument(
        "--skip-lotofoot", action="store_true",
        help="Ne pas scraper les grilles Loto Foot"
    )
    args = parser.parse_args()

    run(
        skip_football_download=args.skip_download,
        skip_football_import=args.skip_import,
        skip_lotofoot=args.skip_lotofoot,
    )
