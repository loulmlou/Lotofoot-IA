"""Scraper des grilles Loto Foot depuis Parions Sport FDJ.

Tente d'abord une approche API JSON, puis fallback vers parsing HTML
avec requests + BeautifulSoup.
"""

import re
import time
import json
import requests
from datetime import datetime
from loguru import logger
from sqlalchemy import select

from config.settings import LOTOFOOT_BASE_URL, LOTOFOOT_TYPES
from database.connection import init_db, SessionLocal
from database.models import GrilleLotoFoot, MatchGrille


# Headers pour simuler un navigateur
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "fr-FR,fr;q=0.9",
}


def build_grille_url(grille_type: str, grille_id: int) -> str:
    """Construit l'URL d'une grille Loto Foot."""
    return f"{LOTOFOOT_BASE_URL}/grilles/loto-foot/{grille_type}/{grille_id}"


def build_api_url(grille_type: str, grille_id: int) -> str:
    """Construit l'URL API potentielle pour une grille."""
    return f"{LOTOFOOT_BASE_URL}/api/grilles/loto-foot/{grille_type}/{grille_id}"


def fetch_grille_json(grille_type: str, grille_id: int, timeout: int = 15) -> dict | None:
    """Tente de récupérer les données d'une grille via l'API JSON.

    Returns:
        Dict des données si trouvé, None sinon.
    """
    # Essayer plusieurs patterns d'API
    api_patterns = [
        build_api_url(grille_type, grille_id),
        f"{LOTOFOOT_BASE_URL}/api/v1/grilles/{grille_type}/{grille_id}",
        f"{LOTOFOOT_BASE_URL}/api/grilles/{grille_id}",
    ]

    for url in api_patterns:
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")
                if "json" in content_type:
                    return response.json()
                # Parfois le JSON est dans une page HTML (données SSR)
                try:
                    return response.json()
                except (json.JSONDecodeError, ValueError):
                    pass
        except requests.RequestException:
            continue

    return None


def fetch_grille_html(grille_type: str, grille_id: int, timeout: int = 15) -> dict | None:
    """Récupère les données d'une grille via parsing HTML.

    Returns:
        Dict avec les données extraites, None si échec.
    """
    from bs4 import BeautifulSoup

    url = build_grille_url(grille_type, grille_id)

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        if response.status_code != 200:
            return None

        # Page trop petite = pas de contenu réel
        if len(response.content) < 1000:
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Détecter la page "grille introuvable"
        h1 = soup.select_one("h1")
        if h1 and "introuvable" in h1.get_text().lower():
            return None

        # Extraction directe depuis le HTML Angular SSR
        grille_data = parse_html_grille(soup, grille_type, grille_id)
        return grille_data

    except requests.RequestException as e:
        logger.debug(f"Erreur HTML pour {grille_type}/{grille_id} : {e}")
        return None


def parse_html_grille(soup, grille_type: str, grille_id: int) -> dict | None:
    """Parse le HTML d'une page de grille Parions Sport (Angular SSR).

    Structure attendue :
    - h1.lotofoot-grid__title → "Résultat Loto Foot 7 N°83"
    - span.lotofoot-grid__date → "fin de valid. 01/07 17h55"
    - div.grid-item → un par match, contient :
      - span.grid-item-team (x2) → noms des équipes
      - input[checked] dans label.grid-item-label → résultat gagnant
    - table.lotosport-ranking-table → rapports et gagnants

    Returns:
        Dict avec numero, date, type, matchs, rapport_rang1, nombre_gagnants.
        None si parsing échoue.
    """
    type_code = LOTOFOOT_TYPES.get(grille_type, {}).get("code", grille_type)

    data = {
        "numero": grille_id,
        "type": type_code,
        "matchs": [],
        "date": None,
        "rapport_rang1": None,
        "nombre_gagnants": None,
    }

    # --- Numéro de grille depuis le titre ---
    title_el = soup.select_one("h1.lotofoot-grid__title")
    if title_el:
        num_match = re.search(r"N\D*(\d+)", title_el.get_text())
        if num_match:
            data["numero_grille"] = int(num_match.group(1))

    # --- Date de la grille ---
    date_el = soup.select_one("span.lotofoot-grid__date, h2.lotofoot-grid__subtitle")
    if date_el:
        date_text = date_el.get_text(strip=True)
        date_match = re.search(r"(\d{2}/\d{2})", date_text)
        if date_match:
            day_month = date_match.group(1)
            # Déduire l'année : utiliser l'année courante par défaut
            year = datetime.now().year
            try:
                data["date"] = datetime.strptime(
                    f"{day_month}/{year}", "%d/%m/%Y"
                ).date()
            except ValueError:
                pass

    # --- Matchs : div.grid-item ---
    grid_items = soup.select("div.grid-item")
    for item in grid_items:
        teams = item.select("span.grid-item-team")
        if len(teams) < 2:
            continue

        domicile = teams[0].get_text(strip=True)
        exterieur = teams[1].get_text(strip=True)

        # Résultat : l'input[checked] dans les labels
        resultat = None
        checked_input = item.select_one("input[checked]")
        if checked_input:
            formcontrol = checked_input.get("formcontrolname", "")
            if formcontrol == "one":
                resultat = "1"
            elif formcontrol == "n":
                resultat = "N"
            elif formcontrol == "two":
                resultat = "2"

        # Fallback : chercher la label avec une classe spécifique
        if resultat is None:
            for label in item.select("label.grid-item-label"):
                inp = label.select_one("input")
                if inp and inp.has_attr("checked"):
                    span = label.select_one("span")
                    if span:
                        val = span.get_text(strip=True)
                        if val in ("1", "N", "2"):
                            resultat = val
                            break

        if domicile and exterieur:
            data["matchs"].append({
                "domicile": domicile,
                "exterieur": exterieur,
                "resultat": resultat,
            })

    # --- Rapports : table.lotosport-ranking-table ---
    ranking_table = soup.select_one("table.lotosport-ranking-table")
    if ranking_table:
        rows = ranking_table.select("tr")
        for row in rows:
            cells = row.select("td")
            if len(cells) >= 2:
                # Première ligne de données = rang 1
                try:
                    gagnants_text = cells[0].get_text(strip=True)
                    montant_text = cells[1].get_text(strip=True)

                    # Extraire le nombre de gagnants
                    gagnants_match = re.search(r"(\d[\d\s]*)", gagnants_text)
                    if gagnants_match:
                        data["nombre_gagnants"] = int(
                            gagnants_match.group(1).replace(" ", "")
                        )

                    # Extraire le montant
                    montant_clean = re.sub(r"[^\d,.]", "", montant_text)
                    montant_clean = montant_clean.replace(",", ".")
                    if montant_clean:
                        data["rapport_rang1"] = float(montant_clean)
                except (ValueError, IndexError):
                    pass
                break  # On ne prend que le rang 1

    if not data["matchs"]:
        return None

    return data


def grille_exists(session, numero: int, type_grille: str) -> bool:
    """Vérifie si une grille existe déjà en base."""
    result = session.execute(
        select(GrilleLotoFoot).where(
            GrilleLotoFoot.id == numero,
            GrilleLotoFoot.type_grille == type_grille,
        )
    ).scalar_one_or_none()
    return result is not None


def save_grille(session, data: dict) -> bool:
    """Sauvegarde une grille et ses matchs en base.

    Returns:
        True si sauvegardée, False si erreur ou déjà existante.
    """
    type_grille = data.get("type", "")
    numero = data.get("numero")

    if not numero or not type_grille:
        return False

    if grille_exists(session, numero, type_grille):
        return False

    date_grille = data.get("date") or datetime.now().date()
    matchs = data.get("matchs", [])

    # Construire la chaîne de résultats
    resultats = "".join(m.get("resultat") or "?" for m in matchs)

    rapport_rang1 = data.get("rapport_rang1")
    nombre_gagnants = data.get("nombre_gagnants")

    grille = GrilleLotoFoot(
        id=numero,
        date=date_grille,
        type_grille=type_grille,
        resultats=resultats,
        rapport_rang1=rapport_rang1,
        nombre_gagnants_rang1=nombre_gagnants,
    )
    session.add(grille)
    session.flush()

    for i, match_data in enumerate(matchs, start=1):
        match_grille = MatchGrille(
            grille_id=grille.id,
            position=i,
            resultat=match_data.get("resultat"),
        )
        session.add(match_grille)

    return True


def scrape_grilles(
    grille_type: str = "loto-foot-7",
    id_start: int | None = None,
    id_end: int = 1,
    delay: float = 1.0,
    max_consecutive_fails: int = 20,
) -> dict:
    """Scrape les grilles d'un type donné, en itérant des IDs en arrière.

    Args:
        grille_type: Type de grille (ex: 'loto-foot-7').
        id_start: ID de départ (le plus récent). Défaut : id_max_approx de la config.
        id_end: ID minimal à atteindre.
        delay: Pause entre les requêtes.
        max_consecutive_fails: Nombre d'échecs consécutifs avant d'arrêter.

    Returns:
        Dict avec les compteurs : scraped, skipped, failed.
    """
    type_info = LOTOFOOT_TYPES.get(grille_type, {})
    if id_start is None:
        id_start = type_info.get("id_max_approx", 100)

    stats = {"scraped": 0, "skipped": 0, "failed": 0}
    consecutive_fails = 0

    init_db()
    session = SessionLocal()

    try:
        for grille_id in range(id_start, id_end - 1, -1):
            type_code = type_info.get("code", grille_type)

            if grille_exists(session, grille_id, type_code):
                stats["skipped"] += 1
                consecutive_fails = 0
                continue

            # Extraction HTML (le site est Angular SSR, pas d'API JSON)
            data = fetch_grille_html(grille_type, grille_id)

            if data is None:
                stats["failed"] += 1
                consecutive_fails += 1
                if consecutive_fails >= max_consecutive_fails:
                    logger.info(
                        f"{max_consecutive_fails} échecs consécutifs pour "
                        f"{grille_type}, arrêt à l'ID {grille_id}"
                    )
                    break
                time.sleep(delay * 0.5)
                continue

            # Normaliser les données
            if "numero" not in data:
                data["numero"] = grille_id
            if "type" not in data:
                data["type"] = type_code

            if save_grille(session, data):
                stats["scraped"] += 1
                logger.debug(f"Grille {grille_type}/{grille_id} sauvegardée")
            else:
                stats["skipped"] += 1

            consecutive_fails = 0
            time.sleep(delay)

        session.commit()

    except Exception as e:
        session.rollback()
        logger.error(f"Erreur scraping {grille_type} : {e}")
    finally:
        session.close()

    logger.info(
        f"Scraping {grille_type} terminé : {stats['scraped']} récupérées, "
        f"{stats['skipped']} ignorées, {stats['failed']} échouées"
    )
    return stats


def scrape_all(delay: float = 1.0, max_consecutive_fails: int = 20) -> dict:
    """Scrape toutes les grilles pour tous les types configurés.

    Returns:
        Dict par type de grille avec les compteurs.
    """
    results = {}
    for grille_type in LOTOFOOT_TYPES:
        logger.info(f"Scraping des grilles {grille_type}...")
        results[grille_type] = scrape_grilles(
            grille_type=grille_type,
            delay=delay,
            max_consecutive_fails=max_consecutive_fails,
        )
    return results


if __name__ == "__main__":
    logger.info("Démarrage du scraping Loto Foot")
    results = scrape_all()
    for gtype, stats in results.items():
        print(f"{gtype} : {stats}")
