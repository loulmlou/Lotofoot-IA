"""Fonctions helpers pour le frontend (sans dépendance Streamlit/Plotly)."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import LOTOFOOT_TYPES

RESULT_COLORS = {"1": "#2ecc71", "N": "#f39c12", "2": "#e74c3c"}

GRID_TYPE_CODES = {v["code"]: v["nb_matchs"] for v in LOTOFOOT_TYPES.values()}


def color_result(result: str) -> str:
    """Retourne un span HTML coloré pour un résultat 1/N/2."""
    color = RESULT_COLORS.get(result, "#888")
    return f'<span style="color:{color};font-weight:bold">{result}</span>'


def format_results_html(resultats: str) -> str:
    """Formate une chaîne de résultats avec couleurs HTML."""
    return " ".join(color_result(c) for c in resultats)
