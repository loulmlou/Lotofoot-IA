"""Fonctions helpers pour le frontend (sans dépendance Streamlit/Plotly)."""

import sys
import os
import re
import unicodedata
from difflib import SequenceMatcher

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import LOTOFOOT_TYPES

RESULT_COLORS = {"1": "#2ecc71", "N": "#f39c12", "2": "#e74c3c"}

GRID_TYPE_CODES = {v["code"]: v["nb_matchs"] for v in LOTOFOOT_TYPES.values()}


def _normalize_name(name: str) -> str:
    """Normalise un nom d'équipe pour la comparaison.

    Supprime accents, met en minuscules, supprime ponctuation et espaces multiples.
    """
    # Suppression des accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Minuscules, suppression ponctuation
    ascii_name = ascii_name.lower()
    ascii_name = re.sub(r"[^a-z0-9\s]", " ", ascii_name)
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip()
    return ascii_name


def match_team_name(fdj_name: str, equipes) -> object | None:
    """Trouve la meilleure correspondance d'une équipe FDJ parmi les équipes en base.

    Args:
        fdj_name: Nom tel qu'affiché sur le site FDJ (ex: "Paris SG").
        equipes: Liste d'objets Equipe (avec attribut .nom).

    Returns:
        L'objet Equipe le plus proche, ou None si aucun score >= 0.65.
    """
    if not fdj_name or not equipes:
        return None

    normalized_fdj = _normalize_name(fdj_name)
    noise_words = {"fc", "sc", "cf", "ac", "as", "us", "rc", "sg",
                   "utd", "de", "du", "le", "la", "les", "of"}
    fdj_words = set(normalized_fdj.split())
    fdj_significant = fdj_words - noise_words

    best_match = None
    best_score = 0.0

    for equipe in equipes:
        normalized_db = _normalize_name(equipe.nom)

        # Match exact normalisé
        if normalized_fdj == normalized_db:
            return equipe

        db_words = set(normalized_db.split())
        db_significant = db_words - noise_words

        # Si tous les mots significatifs du nom FDJ sont dans le nom DB
        # (ex: "Marseille" → "Olympique de Marseille", "Paris SG" → "Paris Saint Germain")
        # C'est un signal fort : l'utilisateur cherche bien cette équipe.
        if fdj_significant and fdj_significant <= db_significant:
            score = 0.85
        # Sens inverse (DB ⊆ FDJ) : exiger une couverture > 50%
        # pour éviter "Inter" (DB) ⊂ "Inter Miami" (FDJ)
        elif db_significant and db_significant <= fdj_significant:
            ratio = len(db_significant) / len(fdj_significant)
            score = 0.85 if ratio > 0.5 else 0.85 * ratio
        else:
            common = fdj_significant & db_significant
            base_score = SequenceMatcher(None, normalized_fdj, normalized_db).ratio()

            if common:
                # Pénaliser si beaucoup de mots non communs
                coverage = len(common) / max(len(fdj_significant | db_significant), 1)
                score = base_score * 0.5 + coverage * 0.5
            else:
                # Aucun mot significatif en commun : exiger un score très élevé
                # pour éviter les faux positifs comme "Houston"→"Luton"
                score = base_score * 0.7

        if score > best_score:
            best_score = score
            best_match = equipe

    if best_score >= 0.65:
        return best_match
    return None


def color_result(result: str) -> str:
    """Retourne un span HTML coloré pour un résultat 1/N/2."""
    color = RESULT_COLORS.get(result, "#888")
    return f'<span style="color:{color};font-weight:bold">{result}</span>'


def format_results_html(resultats: str) -> str:
    """Formate une chaîne de résultats avec couleurs HTML."""
    return " ".join(color_result(c) for c in resultats)
