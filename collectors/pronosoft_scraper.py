"""Scraper des grilles Loto Foot depuis Pronosoft.

Récupère toutes les prochaines grilles (LF7, LF8, LF12, LF15) d'un coup
depuis https://www.pronosoft.com/fr/lotofoot/prochaines-grilles.htm
"""

import re
import requests
from datetime import datetime
from loguru import logger


PRONOSOFT_URL = "https://www.pronosoft.com/fr/lotofoot/prochaines-grilles.htm"

COMBINAISONS_URL_TEMPLATE = (
    "https://www.pronosoft.com/fr/lotosports/combinaisons/loto-foot-{n}/"
)

# Mapping grid_type -> URL slug number
_GRID_TYPE_TO_N = {"LF7": "7", "LF8": "8", "LF12": "12", "LF15": "15"}

# Cache mémoire pour ne pas re-fetcher les stats combinaisons
_combinaisons_cache: dict[str, dict[str, float]] = {}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html, */*",
    "Accept-Language": "fr-FR,fr;q=0.9",
}

# Mapping des types reconnus vers le code interne
_KNOWN_TYPES = {"LF 7": "LF7", "LF 8": "LF8", "LF 12": "LF12", "LF 15": "LF15"}


def _parse_grille_header(header_text: str) -> list[dict]:
    """Parse un header de grille pour en extraire type(s) et numéro.

    Formats possibles :
    - "LF 15 n°57"        → [{"type": "LF15", "numero": 57}]
    - "LF 7 - LF 12"     → [{"type": "LF7", "numero": None}, {"type": "LF12", "numero": None}]
    - "LF 7 n°92"         → [{"type": "LF7", "numero": 92}]
    - "LF 15"             → [{"type": "LF15", "numero": None}]

    Returns:
        Liste de dicts {type, numero} pour chaque type trouvé dans le header.
    """
    results = []

    # Chercher le numéro global (n°XX)
    num_match = re.search(r"n\D*(\d+)", header_text)
    numero = int(num_match.group(1)) if num_match else None

    # Trouver tous les types LF dans le header
    for pattern, code in _KNOWN_TYPES.items():
        if pattern in header_text:
            results.append({"type": code, "numero": numero})

    return results


def _parse_date_from_span(span_tag) -> datetime | None:
    """Extrait la date depuis un <span data-date-utc="...">."""
    if span_tag is None:
        return None
    date_utc = span_tag.get("data-date-utc", "")
    if date_utc:
        try:
            return datetime.strptime(date_utc, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    # Fallback : parser le texte affiché
    text = span_tag.get_text(strip=True)
    date_match = re.search(r"(\d{1,2})\s+(\w+)\s+.?\s*(\d{1,2})h(\d{2})", text)
    if date_match:
        day = int(date_match.group(1))
        month_name = date_match.group(2).lower()
        months = {
            "janvier": 1, "février": 2, "mars": 3, "avril": 4,
            "mai": 5, "juin": 6, "juillet": 7, "août": 8,
            "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
        }
        month = months.get(month_name)
        if month:
            year = datetime.now().year
            try:
                return datetime(year, month, day)
            except ValueError:
                pass
    return None


def _parse_matchs_from_table(table) -> list[dict]:
    """Extrait la liste des matchs depuis une table.grilles-lf."""
    matchs = []
    for row in table.select("tr"):
        cells = row.select("td")
        if len(cells) < 4:
            continue
        home_td = row.select_one("td.equipe.home")
        ext_td = row.select_one("td.equipe.ext")
        if home_td and ext_td:
            domicile = home_td.get_text(strip=True)
            exterieur = ext_td.get_text(strip=True)
            if domicile and exterieur:
                matchs.append({"domicile": domicile, "exterieur": exterieur})
    return matchs


def parse_pronosoft_html(html: str) -> list[dict]:
    """Parse le HTML de la page Pronosoft et retourne les grilles.

    Returns:
        Liste de dicts : {type, numero, date, matchs: [{domicile, exterieur}]}
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    grilles = []

    for table in soup.select("table.grilles-lf"):
        tbody = table.select_one("tbody") or table

        # Header : premier <th colspan="4"> contient le type/numéro
        header_th = tbody.select_one("tr > th[colspan]")
        if not header_th:
            continue
        header_text = header_th.get_text(strip=True)
        type_infos = _parse_grille_header(header_text)
        if not type_infos:
            continue

        # Date : <tr class="valid"> contient un <span data-date-utc="...">
        valid_row = tbody.select_one("tr.valid")
        grille_date = None
        if valid_row:
            span = valid_row.select_one("span[data-date-utc]")
            grille_date = _parse_date_from_span(span)

        # Matchs
        matchs = _parse_matchs_from_table(tbody)

        if not matchs:
            continue

        # Une table peut contenir plusieurs types (ex: "LF 7 - LF 12")
        for info in type_infos:
            nb_matchs_expected = {"LF7": 7, "LF8": 8, "LF12": 12, "LF15": 15}.get(info["type"])
            if nb_matchs_expected and len(matchs) >= nb_matchs_expected:
                grille_matchs = matchs[:nb_matchs_expected]
            else:
                grille_matchs = matchs

            grilles.append({
                "type": info["type"],
                "numero": info["numero"],
                "date": grille_date.date() if grille_date else None,
                "matchs": grille_matchs,
            })

    return grilles


def fetch_upcoming_grilles_pronosoft(timeout: int = 15) -> list[dict]:
    """Récupère toutes les prochaines grilles Loto Foot depuis Pronosoft.

    Returns:
        Liste de dicts : {type, numero, date, matchs: [{domicile, exterieur}]}
        Liste vide en cas d'erreur réseau.
    """
    try:
        response = requests.get(PRONOSOFT_URL, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return parse_pronosoft_html(response.text)
    except requests.RequestException as e:
        logger.error(f"Erreur lors de la récupération Pronosoft : {e}")
        return []


def parse_combinaisons_html(html: str) -> dict[str, int]:
    """Parse le HTML d'une page combinaisons Pronosoft.

    Cherche la table.stat_1n2 dont le header contient "Combinaisons 1N2"
    et extrait chaque profil avec son nombre de sorties.

    Returns:
        dict {"3-2-2": 532, "4-1-2": 459, ...}
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, int] = {}

    for table in soup.select("table.stat_1n2"):
        # Vérifier que c'est la table "Combinaisons 1N2"
        header = table.select_one("th")
        if not header or "Combinaisons 1N2" not in header.get_text():
            continue

        for row in table.select("tr"):
            cells = row.select("td")
            if len(cells) < 4:
                continue

            try:
                nb1 = int(cells[0].get_text(strip=True))
                nb_n = int(cells[1].get_text(strip=True))
                nb2 = int(cells[2].get_text(strip=True))
            except (ValueError, IndexError):
                continue

            # Colonne 3 : "532 fois"
            sorties_text = cells[3].get_text(strip=True)
            sorties_match = re.search(r"(\d+)", sorties_text)
            if not sorties_match:
                continue

            sorties = int(sorties_match.group(1))
            key = f"{nb1}-{nb_n}-{nb2}"
            result[key] = sorties

        # On a trouvé la bonne table, pas besoin de continuer
        break

    return result


def fetch_combinaisons_stats(grid_type: str) -> dict[str, float]:
    """Fetch et normalise les stats combinaisons 1N2 pour un type de grille.

    Args:
        grid_type: "LF7", "LF8", "LF12" ou "LF15"

    Returns:
        dict {"3-2-2": 0.108, ...} normalisé en proportions.
        Dict vide si le type est inconnu ou en cas d'erreur réseau.
    """
    if grid_type in _combinaisons_cache:
        return _combinaisons_cache[grid_type]

    n = _GRID_TYPE_TO_N.get(grid_type)
    if not n:
        logger.warning(f"Type de grille inconnu pour combinaisons: {grid_type}")
        return {}

    url = COMBINAISONS_URL_TEMPLATE.format(n=n)
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Erreur fetch combinaisons {grid_type}: {e}")
        return {}

    counts = parse_combinaisons_html(response.text)
    if not counts:
        logger.warning(f"Aucune combinaison trouvée pour {grid_type}")
        return {}

    total = sum(counts.values())
    if total == 0:
        return {}

    stats = {k: v / total for k, v in counts.items()}
    _combinaisons_cache[grid_type] = stats
    return stats


if __name__ == "__main__":
    import json

    grilles = fetch_upcoming_grilles_pronosoft()
    for g in grilles:
        print(json.dumps(
            {k: str(v) if k == "date" else v for k, v in g.items()},
            ensure_ascii=False, indent=2,
        ))
