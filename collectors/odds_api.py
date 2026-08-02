"""Collecteur de cotes via The Odds API (fallback quand Pronosoft manque).

API gratuite: 500 requetes/mois sur the-odds-api.com
Docs: https://the-odds-api.com/liveapi/guides/v4/

Usage:
    from collectors.odds_api import fetch_odds_for_matches
    matches = fetch_odds_for_matches(matches_list)
"""

import os
import re
from difflib import SequenceMatcher

import requests
from loguru import logger


# Cle API via variable d'environnement
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Mapping des ligues LotoFoot vers les sport keys de The Odds API
LEAGUE_KEYS = [
    "soccer_france_ligue_one",
    "soccer_france_ligue_two",
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_netherlands_eredivisie",
    "soccer_portugal_primeira_liga",
    "soccer_belgium_first_div",
    "soccer_turkey_super_league",
    "soccer_usa_mls",
    "soccer_brazil_serie_a",
    "soccer_mexico_ligamx",
    "soccer_japan_j_league",
    "soccer_conmebol_copa_libertadores",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_uefa_europa_conference_league",
]


def _normalize_team(name: str) -> str:
    """Normalise un nom d'equipe pour la comparaison floue."""
    name = name.lower().strip()
    # Supprimer les suffixes courants
    for suffix in ["fc", "sc", "ac", "cf", "rc", "as", "us", "ss", "sg",
                    "athletic", "athletico", "sporting"]:
        name = re.sub(rf"\b{suffix}\b", "", name)
    # Supprimer la ponctuation
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _match_score(team_a: str, team_b: str) -> float:
    """Score de similarite entre deux noms d'equipe."""
    a = _normalize_team(team_a)
    b = _normalize_team(team_b)
    # Match exact normalise
    if a == b:
        return 1.0
    # Inclusion
    if a in b or b in a:
        return 0.85
    # Similarite de sequence
    return SequenceMatcher(None, a, b).ratio()


def fetch_upcoming_odds(sport_key: str, regions: str = "eu",
                        markets: str = "h2h") -> list[dict]:
    """Recupere les cotes a venir pour un sport.

    Args:
        sport_key: cle du sport (ex: "soccer_france_ligue_one")
        regions: region des bookmakers ("eu", "uk", "us")
        markets: marches ("h2h" = 1X2)

    Returns:
        liste de matchs avec cotes, ou [] en cas d'erreur
    """
    if not ODDS_API_KEY:
        logger.warning("ODDS_API_KEY non definie. Definir la variable d'environnement.")
        return []

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"Erreur The Odds API ({sport_key}): {e}")
        return []


def _extract_best_odds(event: dict) -> dict:
    """Extrait les meilleures cotes 1X2 d'un event The Odds API.

    Prend la moyenne des cotes de tous les bookmakers pour lisser.

    Returns:
        dict {home, away, cote_1, cote_n, cote_2}
    """
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")

    cotes_1 = []
    cotes_n = []
    cotes_2 = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            if home_team in outcomes:
                cotes_1.append(outcomes[home_team])
            if "Draw" in outcomes:
                cotes_n.append(outcomes["Draw"])
            if away_team in outcomes:
                cotes_2.append(outcomes[away_team])

    result = {
        "home": home_team,
        "away": away_team,
        "cote_1": round(sum(cotes_1) / len(cotes_1), 2) if cotes_1 else 0,
        "cote_n": round(sum(cotes_n) / len(cotes_n), 2) if cotes_n else 0,
        "cote_2": round(sum(cotes_2) / len(cotes_2), 2) if cotes_2 else 0,
    }
    return result


def fetch_odds_for_matches(matches: list[dict]) -> list[dict]:
    """Complete les cotes manquantes des matchs via The Odds API.

    Pour chaque match sans cotes (cote_1/cote_n/cote_2 a 0), cherche
    le match correspondant dans The Odds API par matching flou des equipes.

    Args:
        matches: liste de dicts avec home, away, cote_1, cote_n, cote_2

    Returns:
        la meme liste avec les cotes completees
    """
    if not ODDS_API_KEY:
        logger.warning("ODDS_API_KEY non definie, impossible de completer les cotes.")
        return matches

    # Identifier les matchs sans cotes
    missing = []
    for i, m in enumerate(matches):
        if not m.get("cote_1") or not m.get("cote_n") or not m.get("cote_2"):
            missing.append(i)

    if not missing:
        return matches

    logger.info(f"{len(missing)} match(s) sans cotes, interrogation The Odds API...")

    # Recuperer les cotes de toutes les ligues
    all_events = []
    for league_key in LEAGUE_KEYS:
        events = fetch_upcoming_odds(league_key)
        all_events.extend(events)
        if events:
            logger.info(f"  {league_key}: {len(events)} matchs")

    if not all_events:
        logger.warning("Aucun match trouve via The Odds API.")
        return matches

    # Extraire les cotes de chaque event
    api_odds = [_extract_best_odds(e) for e in all_events]

    # Matcher les matchs manquants
    filled = 0
    for idx in missing:
        m = matches[idx]
        home = m.get("home", "")
        away = m.get("away", "")

        if not home or not away:
            continue

        best_match = None
        best_score = 0.0

        for odds in api_odds:
            # Score = moyenne du matching home + away
            score_h = _match_score(home, odds["home"])
            score_a = _match_score(away, odds["away"])
            score = (score_h + score_a) / 2

            if score > best_score:
                best_score = score
                best_match = odds

        # Seuil de matching : 0.55 minimum
        if best_match and best_score >= 0.55:
            if best_match["cote_1"] > 0 and best_match["cote_n"] > 0 and best_match["cote_2"] > 0:
                m["cote_1"] = best_match["cote_1"]
                m["cote_n"] = best_match["cote_n"]
                m["cote_2"] = best_match["cote_2"]
                m["odds_source"] = "the-odds-api"
                filled += 1
                logger.info(
                    f"  Match {home} vs {away} -> "
                    f"{best_match['home']} vs {best_match['away']} "
                    f"(score={best_score:.2f}) : "
                    f"1={m['cote_1']} N={m['cote_n']} 2={m['cote_2']}"
                )

    logger.info(f"Cotes completees: {filled}/{len(missing)} matchs")
    return matches
