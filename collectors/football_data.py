"""Collecteur de données historiques depuis football-data.co.uk.

Télécharge les fichiers CSV pour toutes les ligues et saisons configurées.
"""

import os
import time
import requests
from loguru import logger

from config.settings import (
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_LEAGUES,
    FOOTBALL_DATA_SEASONS,
    RAW_FOOTBALL_DIR,
)


def build_url(season: str, league_code: str) -> str:
    """Construit l'URL de téléchargement pour une saison et une ligue."""
    return f"{FOOTBALL_DATA_BASE_URL}/{season}/{league_code}.csv"


def download_csv(url: str, dest_path: str, timeout: int = 30) -> bool:
    """Télécharge un fichier CSV depuis une URL.

    Returns:
        True si le téléchargement a réussi, False sinon.
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200 and len(response.content) > 100:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(response.content)
            logger.debug(f"Téléchargé : {dest_path}")
            return True
        else:
            logger.warning(
                f"Échec {url} (status={response.status_code}, "
                f"taille={len(response.content)})"
            )
            return False
    except requests.RequestException as e:
        logger.warning(f"Erreur réseau pour {url} : {e}")
        return False


def get_csv_path(league_code: str, season: str) -> str:
    """Retourne le chemin local pour un CSV donné."""
    return os.path.join(RAW_FOOTBALL_DIR, f"{league_code}_{season}.csv")


def collect_all(
    leagues: dict | None = None,
    seasons: list[str] | None = None,
    delay: float = 0.5,
    skip_existing: bool = True,
) -> dict:
    """Télécharge tous les CSV pour les ligues et saisons configurées.

    Args:
        leagues: Dict des ligues (défaut : FOOTBALL_DATA_LEAGUES).
        seasons: Liste des saisons (défaut : FOOTBALL_DATA_SEASONS).
        delay: Pause entre les requêtes (en secondes).
        skip_existing: Ne pas re-télécharger les fichiers existants.

    Returns:
        Dict avec les compteurs : downloaded, skipped, failed.
    """
    leagues = leagues or FOOTBALL_DATA_LEAGUES
    seasons = seasons or FOOTBALL_DATA_SEASONS

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    os.makedirs(RAW_FOOTBALL_DIR, exist_ok=True)

    total = len(leagues) * len(seasons)
    current = 0

    for league_code in leagues:
        for season in seasons:
            current += 1
            dest = get_csv_path(league_code, season)

            if skip_existing and os.path.exists(dest):
                stats["skipped"] += 1
                continue

            url = build_url(season, league_code)
            logger.info(f"[{current}/{total}] Téléchargement {league_code} {season}...")

            if download_csv(url, dest):
                stats["downloaded"] += 1
            else:
                stats["failed"] += 1

            time.sleep(delay)

    logger.info(
        f"Collecte terminée : {stats['downloaded']} téléchargés, "
        f"{stats['skipped']} ignorés, {stats['failed']} échoués"
    )
    return stats


if __name__ == "__main__":
    logger.info("Démarrage de la collecte football-data.co.uk")
    result = collect_all()
    print(f"Résultat : {result}")
