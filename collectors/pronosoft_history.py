"""Scraper historique des grilles Loto Foot depuis Pronosoft.

Récupère les données de répartition (cotes, pourcentages joueurs, prono Cyborg)
et les résultats historiques pour chaque grille.
Supporte LF7, LF8, LF12, LF15.
"""

import json
import os
import re
import time
from statistics import mean, median, stdev

import requests
from bs4 import BeautifulSoup
from loguru import logger


PRONOSOFT_BASE = "https://www.pronosoft.com/fr"

REPARTITION_URL_TPL = (
    PRONOSOFT_BASE + "/lotosports/repartition/{type_slug}/{year}-grille-{n}/"
)
HISTORIQUE_URL_TPL = (
    PRONOSOFT_BASE + "/lotosports/historiques/{type_slug}/{season}/"
    "{year}-grille-{n}/"
)

# Saisons disponibles par type: (season_label, year, nb_grilles_approx)
AVAILABLE_SEASONS = {
    "LF7": [
        ("2025-2026", 2026, 94),
        ("2024-2025", 2025, 91),
        ("2023-2024", 2024, 91),
        ("2022-2023", 2023, 91),
    ],
    "LF8": [
        ("2025-2026", 2026, 94),
        ("2024-2025", 2025, 91),
        ("2023-2024", 2024, 91),
        ("2022-2023", 2023, 91),
    ],
    "LF12": [
        ("2025-2026", 2026, 60),
        ("2024-2025", 2025, 55),
        ("2023-2024", 2024, 55),
        ("2022-2023", 2023, 55),
    ],
    "LF15": [
        ("2025-2026", 2026, 94),
        ("2024-2025", 2025, 91),
        ("2023-2024", 2024, 91),
        ("2022-2023", 2023, 91),
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html, */*",
    "Accept-Language": "fr-FR,fr;q=0.9",
}

DEFAULT_HISTORY_PATH = os.path.join("data", "history", "lf7_history.json")


# ---------------------------------------------------------------------------
# Helpers multi-type
# ---------------------------------------------------------------------------

def _slugs_for_type(grid_type: str) -> tuple[str, str, int]:
    """Retourne (slug, type_slug, nb_matchs) pour un type de grille.

    Exemples:
        LF7  -> ('lf7',  'loto-foot-7',  7)
        LF8  -> ('lf8',  'loto-foot-8',  8)
        LF12 -> ('lf12', 'loto-foot-12', 12)
        LF15 -> ('lf15', 'loto-foot-15', 15)
    """
    gt = grid_type.upper()
    nb = int(gt.replace("LF", ""))
    slug = gt.lower()
    type_slug = f"loto-foot-{nb}"
    return slug, type_slug, nb


def get_history_path(grid_type: str = "LF7") -> str:
    """Retourne le chemin du fichier historique pour un type de grille."""
    return os.path.join("data", "history", f"{grid_type.lower()}_history.json")


# ---------------------------------------------------------------------------
# Parsing répartition
# ---------------------------------------------------------------------------

def _parse_cote_cell(text: str) -> tuple[float, float]:
    """Parse une cellule de cote (ex: '17%2.25' ou '64.9%1.35').

    Formats geres :
    - "45%1,52"   -> (45.0, 1.52)   pct + cote
    - "-1,77"     -> (0.0, 1.77)    tiret + cote (pas de pct)
    - "58%"       -> (58.0, 0.0)    pct seul (pas de cote)
    - "2.25"      -> (0.0, 2.25)    cote seule

    Returns:
        (pourcentage_joueurs, cote)
    """
    text = text.strip()
    # Format: "XX%Y.YY" or "XX.X%Y.YY"
    match = re.match(r"([\d.,]+)\s*%\s*([\d.,]+)", text)
    if match:
        pct = float(match.group(1).replace(",", "."))
        cote = float(match.group(2).replace(",", "."))
        return pct, cote

    # Format: "XX%" (pourcentage seul, pas de cote)
    match_pct = re.match(r"([\d.,]+)\s*%\s*$", text)
    if match_pct:
        pct = float(match_pct.group(1).replace(",", "."))
        return pct, 0.0

    # Format: "-X.XX" (tiret + cote, pas de pourcentage)
    match_dash = re.match(r"-\s*([\d.,]+)", text)
    if match_dash:
        cote = float(match_dash.group(1).replace(",", "."))
        return 0.0, cote

    # Fallback: just a number (cote only)
    try:
        cote = float(text.replace(",", "."))
        return 0.0, cote
    except (ValueError, TypeError):
        return 0.0, 0.0


def parse_repartition_html(html: str) -> dict:
    """Parse la page répartition d'une grille LF7.

    Extrait pour chaque match :
    - home, away, cote_1, cote_n, cote_2, pct_1, pct_n, pct_2
    - prono_cyborg, score

    Extrait au niveau grille :
    - difficulty (indice de difficulté)
    - nb_pronostics

    Returns:
        dict avec clés 'matches', 'difficulty', 'nb_pronostics'
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {"matches": [], "difficulty": None, "nb_pronostics": None}

    # --- Indice de difficulté ---
    for h3 in soup.find_all("h3"):
        if "difficulté" in h3.get_text().lower():
            next_p = h3.find_next("p")
            if next_p:
                diff_text = next_p.get_text(strip=True).replace(",", ".")
                diff_match = re.search(r"([\d.]+)", diff_text)
                if diff_match:
                    result["difficulty"] = float(diff_match.group(1))
            break

    # --- Nombre de pronostics ---
    page_text = soup.get_text()
    prono_match = re.search(r"([\d\s]+)\s*pronostics?", page_text, re.IGNORECASE)
    if prono_match:
        nb_text = prono_match.group(1).replace(" ", "").replace("\xa0", "")
        try:
            result["nb_pronostics"] = int(nb_text)
        except ValueError:
            pass

    # --- Table des matchs ---
    table = soup.find("table", class_="prono-cyb-des")
    if not table:
        return result

    position = 0
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # Match cell
        match_td = row.find("td", class_="match")
        if not match_td:
            continue

        match_text = match_td.get_text(strip=True)
        # Split teams: "Team A - Team B" or "Team A–Team B"
        teams = re.split(r"\s*[-–]\s*", match_text, maxsplit=1)
        if len(teams) != 2:
            continue

        home = teams[0].strip()
        away = teams[1].strip()
        position += 1

        # Cote cells (1, N, 2)
        cote_cells = row.find_all("td", class_=re.compile(r"cote-d|cote-bet"))
        pct_1, cote_1 = 0.0, 0.0
        pct_n, cote_n = 0.0, 0.0
        pct_2, cote_2 = 0.0, 0.0

        # Find all cells with cote data - they come in order 1, N, 2
        # Only take the first 3 cote-d/cote-bet cells (ignore "over/under" cols)
        cote_tds = row.find_all("td", class_=re.compile(r"cote"))
        cote_data = []
        for td in cote_tds:
            if len(cote_data) >= 3:
                break
            td_text = td.get_text(strip=True)
            if not td_text or td_text == "-":
                continue
            if "%" in td_text or re.match(r"[-\d.,]+", td_text):
                pct, cote = _parse_cote_cell(td_text)
                cote_data.append((pct, cote))

        if len(cote_data) >= 3:
            pct_1, cote_1 = cote_data[0]
            pct_n, cote_n = cote_data[1]
            pct_2, cote_2 = cote_data[2]

        # Prono Cyborg
        prono_td = row.find("td", class_="prono_match")
        prono_cyborg = prono_td.get_text(strip=True) if prono_td else ""

        # Score
        score_td = row.find("td", class_=re.compile(r"dev_desktop_td_score|score"))
        score = score_td.get_text(strip=True) if score_td else ""

        result["matches"].append({
            "position": position,
            "home": home,
            "away": away,
            "cote_1": cote_1,
            "cote_n": cote_n,
            "cote_2": cote_2,
            "pct_1": pct_1,
            "pct_n": pct_n,
            "pct_2": pct_2,
            "prono_cyborg": prono_cyborg,
            "score": score,
        })

    return result


# ---------------------------------------------------------------------------
# Parsing historique
# ---------------------------------------------------------------------------

def parse_historique_html(html: str, nb_matchs: int = 7) -> dict:
    """Parse la page historique d'une grille.

    Extrait :
    - Résultat de chaque match (span .res)
    - Rapports (gagnants + montant)
    - Stats combinaison (profil, etc.)

    Args:
        html: contenu HTML de la page
        nb_matchs: nombre de matchs de la grille (7, 8, 12, 15)

    Returns:
        dict avec clés 'resultats_par_match', 'resultats_str', 'profil',
        'rapports', 'stats_combinaison'
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "resultats_par_match": [],
        "resultats_str": "",
        "profil": "",
        "rapports": {},
        "stats_combinaison": {},
    }

    # --- Résultats des matchs (table .hist) ---
    hist_table = soup.find("table", class_="hist")
    if hist_table:
        for row in hist_table.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue

            res_span = row.find("span", class_="res")
            if res_span:
                resultat = res_span.get_text(strip=True)
                result["resultats_par_match"].append(resultat)

    result["resultats_str"] = "".join(result["resultats_par_match"])

    # Compute profil from results
    if result["resultats_str"]:
        r = result["resultats_str"]
        nb1 = r.count("1")
        nb_n = r.count("N")
        nb2 = r.count("2")
        result["profil"] = f"{nb1}-{nb_n}-{nb2}"

    # --- Rapports ---
    # Structure HTML: <div class="rapports"><h2>Rapports Officiels...</h2>
    #   <table><tr><td>7</td><td>128</td><td>781,0 €</td></tr>
    #          <tr><td>6</td><td>2 051</td><td>52,0 €</td></tr></table>
    rapports_div = soup.find("div", class_="rapports")
    if rapports_div:
        rap_table = rapports_div.find("table")
        if rap_table:
            for row in rap_table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue
                try:
                    rang = int(cells[0].get_text(strip=True))
                except (ValueError, IndexError):
                    continue

                rang_key = f"{rang}_sur_{nb_matchs}"

                # Gagnants (2nd cell)
                gagnants = 0
                g_text = cells[1].get_text(strip=True).replace("\xa0", "").replace(" ", "")
                try:
                    gagnants = int(g_text)
                except ValueError:
                    pass

                # Montant (3rd cell, e.g. "781,0 €")
                montant = 0.0
                m_text = cells[2].get_text(strip=True)
                m_match = re.search(r"([\d\s\xa0]+[.,]\d+)", m_text)
                if m_match:
                    m_str = m_match.group(1).replace(" ", "").replace("\xa0", "").replace(",", ".")
                    try:
                        montant = float(m_str)
                    except ValueError:
                        pass

                result["rapports"][rang_key] = {
                    "gagnants": gagnants,
                    "montant": montant,
                }
    else:
        # Fallback: look for tables with rapport-like data
        for table in soup.find_all("table"):
            text = table.get_text()
            if "gagnant" in text.lower() and ("rapport" in text.lower() or "\u20ac" in text):
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) < 2:
                        continue
                    rang_text = cells[0].get_text(strip=True)
                    rang_match = re.search(r"(\d+)\s*(?:sur|/)\s*(\d+)", rang_text)
                    if not rang_match:
                        continue
                    rang_key = f"{rang_match.group(1)}_sur_{rang_match.group(2)}"
                    gagnants = 0
                    montant = 0.0
                    for cell in cells[1:]:
                        cell_text = cell.get_text(strip=True)
                        montant_match = re.search(r"([\d\s]+[.,]\d+)\s*\u20ac?", cell_text)
                        if montant_match:
                            m_str = montant_match.group(1).replace(" ", "").replace(",", ".")
                            try:
                                montant = float(m_str)
                            except ValueError:
                                pass
                        else:
                            g_match = re.search(r"([\d\s]+)", cell_text)
                            if g_match:
                                try:
                                    gagnants = int(g_match.group(1).replace(" ", ""))
                                except ValueError:
                                    pass
                    result["rapports"][rang_key] = {
                        "gagnants": gagnants,
                        "montant": montant,
                    }

    # --- Stats combinaison (table .rapports) ---
    rapports_table = soup.find("table", class_="rapports")
    if rapports_table:
        for row in rapports_table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).lower()
                val = cells[1].get_text(strip=True)
                result["stats_combinaison"][key] = val

    return result


# ---------------------------------------------------------------------------
# Fetching & merging
# ---------------------------------------------------------------------------

def fetch_grid_history(grid_number: int, year: int = 2026,
                       season: str | None = None,
                       grid_type: str = "LF7",
                       timeout: int = 15) -> dict | None:
    """Fetch répartition + historique pour une grille, merge, calcule métriques.

    Args:
        grid_number: numéro de la grille (1-91+)
        year: année de la grille
        season: label de la saison (ex: '2025-2026'). Auto-detecté si None.
        grid_type: type de grille (LF7, LF8, LF12, LF15)
        timeout: timeout HTTP en secondes

    Returns:
        dict complet de la grille, ou None en cas d'erreur
    """
    slug, type_slug, nb_matchs = _slugs_for_type(grid_type)

    if season is None:
        season = f"{year - 1}-{year}"

    grid = {
        "grid_number": grid_number,
        "year": year,
        "season": season,
        "grid_type": grid_type.upper(),
        "nb_matchs": nb_matchs,
        "matches": [],
    }

    # 1. Fetch répartition
    rep_url = REPARTITION_URL_TPL.format(type_slug=type_slug, year=year, n=grid_number)
    try:
        resp = requests.get(rep_url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        rep_data = parse_repartition_html(resp.text)
        grid["matches"] = rep_data["matches"]
        grid["difficulty"] = rep_data.get("difficulty")
        grid["nb_pronostics"] = rep_data.get("nb_pronostics")
    except requests.RequestException as e:
        logger.warning(f"Erreur fetch répartition grille {grid_type} {grid_number}: {e}")
        return None

    # 2. Fetch historique
    hist_url = HISTORIQUE_URL_TPL.format(
        type_slug=type_slug, season=season, year=year, n=grid_number,
    )
    try:
        resp = requests.get(hist_url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        hist_data = parse_historique_html(resp.text, nb_matchs=nb_matchs)
        grid["resultats"] = hist_data.get("resultats_str", "")
        grid["profil"] = hist_data.get("profil", "")
        grid["rapports"] = hist_data.get("rapports", {})
        grid["stats_combinaison"] = hist_data.get("stats_combinaison", {})

        # Merge résultats into matches
        resultats = hist_data.get("resultats_par_match", [])
        for i, match in enumerate(grid["matches"]):
            if i < len(resultats):
                match["resultat"] = resultats[i]
    except requests.RequestException as e:
        logger.warning(f"Erreur fetch historique grille {grid_type} {grid_number}: {e}")
        # Continue without historique data

    # 3. Compute metrics
    metrics = compute_grid_metrics(grid)
    grid.update(metrics)

    return grid


def fetch_all_history(start: int = 1, end: int = 91,
                      year: int = 2026, grid_type: str = "LF7",
                      delay: float = 1.0) -> list[dict]:
    """Scrape toutes les grilles d'une saison avec rate limiting.

    Args:
        start: premier numéro de grille
        end: dernier numéro de grille (inclus)
        year: année de la saison
        grid_type: type de grille (LF7, LF8, LF12, LF15)
        delay: délai entre les requêtes en secondes

    Returns:
        liste de dicts (grilles valides uniquement)
    """
    season = f"{year - 1}-{year}"
    grids = []
    for n in range(start, end + 1):
        logger.info(f"Scraping {grid_type} grille {n}/{end} (saison {season})...")
        grid = fetch_grid_history(n, year=year, season=season, grid_type=grid_type)
        if grid:
            grids.append(grid)
            logger.info(
                f"  Grille {n}: {len(grid.get('matches', []))} matchs, "
                f"difficulté={grid.get('difficulty')}, "
                f"résultats={grid.get('resultats', '?')}"
            )
        else:
            logger.warning(f"  Grille {n}: échec")

        if n < end:
            time.sleep(delay)

    logger.info(f"Scraping {grid_type} terminé: {len(grids)}/{end - start + 1} grilles récupérées")
    return grids


# Backward compat alias
fetch_all_lf7_history = fetch_all_history


def fetch_multi_season_history(seasons: list[tuple] | None = None,
                               grid_type: str = "LF7",
                               delay: float = 1.5) -> list[dict]:
    """Scrape les grilles sur plusieurs saisons.

    Args:
        seasons: liste de (season_label, year, nb_grilles).
                 Si None, utilise AVAILABLE_SEASONS pour le grid_type.
        grid_type: type de grille (LF7, LF8, LF12, LF15)
        delay: délai entre les requêtes en secondes

    Returns:
        liste de dicts (toutes saisons confondues, triée par saison puis numero)
    """
    if seasons is None:
        seasons = AVAILABLE_SEASONS.get(grid_type.upper(), AVAILABLE_SEASONS["LF7"])

    all_grids = []
    for season_label, year, nb_grilles in seasons:
        logger.info(f"=== Saison {season_label} (year={year}, ~{nb_grilles} grilles) ===")
        prev_resultats = None
        consecutive_dupes = 0

        for n in range(1, nb_grilles + 1):
            logger.info(f"Scraping {season_label} grille {n}/{nb_grilles}...")
            grid = fetch_grid_history(n, year=year, season=season_label,
                                     grid_type=grid_type, timeout=30)

            if not grid:
                logger.warning(f"  Grille {n}: echec")
                time.sleep(delay)
                continue

            # Detection de redirection Pronosoft (grilles inexistantes)
            resultats = grid.get("resultats", "")
            if resultats and resultats == prev_resultats:
                consecutive_dupes += 1
                if consecutive_dupes >= 2:
                    logger.info(f"  Grille {n}: doublon detecte, arret de la saison.")
                    break
            else:
                consecutive_dupes = 0
            prev_resultats = resultats

            # Ne garder que les grilles avec des resultats
            if resultats and grid.get("matches"):
                all_grids.append(grid)
                logger.info(
                    f"  Grille {n}: {len(grid.get('matches', []))} matchs, "
                    f"resultats={resultats}"
                )

            if n < nb_grilles:
                time.sleep(delay)

    logger.info(f"Multi-saison {grid_type} termine: {len(all_grids)} grilles recuperees")
    return all_grids


# ---------------------------------------------------------------------------
# Métriques de grille
# ---------------------------------------------------------------------------

def compute_grid_metrics(grid: dict) -> dict:
    """Calcule les métriques dérivées d'une grille à partir des cotes.

    Args:
        grid: dict avec clé 'matches' contenant les données de cotes

    Returns:
        dict avec somme_cotes_fav, moyenne_cote_fav, mediane_cote_fav,
        ecart_type_cotes, nb_surprises
    """
    matches = grid.get("matches", [])
    if not matches:
        return {
            "somme_cotes_fav": 0.0,
            "moyenne_cote_fav": 0.0,
            "mediane_cote_fav": 0.0,
            "ecart_type_cotes": 0.0,
            "nb_surprises": 0,
        }

    cotes_fav = []
    nb_surprises = 0

    for match in matches:
        cotes = [
            match.get("cote_1", 0),
            match.get("cote_n", 0),
            match.get("cote_2", 0),
        ]
        # Filter out zero/invalid cotes
        valid_cotes = [c for c in cotes if c and c > 0]
        if valid_cotes:
            fav_cote = min(valid_cotes)
            cotes_fav.append(fav_cote)

            # Count surprises: result != favorite
            resultat = match.get("resultat", "")
            if resultat and valid_cotes:
                min_idx = cotes.index(min(valid_cotes))
                fav_result = ["1", "N", "2"][min_idx]
                if resultat != fav_result:
                    nb_surprises += 1

    if not cotes_fav:
        return {
            "somme_cotes_fav": 0.0,
            "moyenne_cote_fav": 0.0,
            "mediane_cote_fav": 0.0,
            "ecart_type_cotes": 0.0,
            "nb_surprises": 0,
        }

    # Indicateurs avances (issus de l'analyse de correlations)
    cotes_outsider = []
    spreads = []
    nb_matchs_serres = 0
    nb_matchs_faciles = 0
    accord_count = 0
    accord_total = 0

    for match in matches:
        c = [
            match.get("cote_1", 0),
            match.get("cote_n", 0),
            match.get("cote_2", 0),
        ]
        valid = [x for x in c if x and x > 0]
        if valid:
            fav = min(valid)
            outsider = max(valid)
            cotes_outsider.append(outsider)
            spreads.append(outsider - fav)
            if fav > 2.0:
                nb_matchs_serres += 1
            if fav < 1.5:
                nb_matchs_faciles += 1

            # Accord cotes/joueurs
            p = [match.get("pct_1", 0), match.get("pct_n", 0), match.get("pct_2", 0)]
            if any(x > 0 for x in p):
                fav_idx = c.index(fav)
                plus_joue_idx = p.index(max(p))
                accord_total += 1
                if fav_idx == plus_joue_idx:
                    accord_count += 1

    # inv_spread_sum: somme des inverses des spreads (plus c'est haut = plus c'est serre)
    inv_spread_sum = sum(1 / s if s > 0 else 10 for s in spreads)

    return {
        "somme_cotes_fav": round(sum(cotes_fav), 2),
        "moyenne_cote_fav": round(mean(cotes_fav), 2),
        "mediane_cote_fav": round(median(cotes_fav), 2),
        "ecart_type_cotes": round(stdev(cotes_fav), 2) if len(cotes_fav) > 1 else 0.0,
        "nb_surprises": nb_surprises,
        # Nouveaux indicateurs (correlations fortes avec le resultat)
        "std_cote_fav": round(stdev(cotes_fav), 4) if len(cotes_fav) > 1 else 0.0,
        "inv_spread_sum": round(inv_spread_sum, 4),
        "nb_matchs_serres": nb_matchs_serres,
        "nb_matchs_faciles": nb_matchs_faciles,
        "accord_cotes_joueurs": round(accord_count / accord_total, 4) if accord_total > 0 else 0.0,
        "coeff_variation_fav": round((stdev(cotes_fav) / mean(cotes_fav)), 4) if len(cotes_fav) > 1 and mean(cotes_fav) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_history(grids: list, path: str | None = None,
                 grid_type: str = "LF7") -> None:
    """Sauvegarde l'historique des grilles en JSON.

    Args:
        grids: liste de dicts de grilles
        path: chemin du fichier JSON (auto-detect depuis grid_type si None)
        grid_type: type de grille (utilisé si path est None)
    """
    if path is None:
        path = get_history_path(grid_type)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(grids, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Historique sauvegardé: {len(grids)} grilles -> {path}")


def load_history(path: str | None = None, grid_type: str = "LF7") -> list:
    """Charge l'historique des grilles depuis un fichier JSON.

    Args:
        path: chemin du fichier JSON (auto-detect depuis grid_type si None)
        grid_type: type de grille (utilisé si path est None)

    Returns:
        liste de dicts de grilles, ou liste vide si le fichier n'existe pas
    """
    if path is None:
        path = get_history_path(grid_type)
    if not os.path.exists(path):
        logger.warning(f"Fichier historique non trouvé: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Historique chargé: {len(data)} grilles depuis {path}")
    return data


if __name__ == "__main__":
    import sys

    gt = sys.argv[2].upper() if len(sys.argv) > 2 else "LF7"
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        grid = fetch_grid_history(n, grid_type=gt)
        if grid:
            print(json.dumps(grid, ensure_ascii=False, indent=2, default=str))
    else:
        grids = fetch_all_history(grid_type=gt)
        save_history(grids, grid_type=gt)
