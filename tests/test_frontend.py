"""Tests pour le frontend Streamlit."""

import sys
import os

import pytest

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_import_helpers():
    """Vérifie que l'import du module helpers ne crashe pas."""
    from frontend import helpers  # noqa: F401


def test_color_result():
    """Vérifie le formatage HTML des résultats."""
    from frontend.helpers import color_result

    html_1 = color_result("1")
    assert "2ecc71" in html_1
    assert ">1<" in html_1

    html_n = color_result("N")
    assert "f39c12" in html_n
    assert ">N<" in html_n

    html_2 = color_result("2")
    assert "e74c3c" in html_2
    assert ">2<" in html_2


def test_color_result_unknown():
    """Vérifie le fallback pour un résultat inconnu."""
    from frontend.helpers import color_result

    html = color_result("X")
    assert "#888" in html
    assert ">X<" in html


def test_format_results_html():
    """Vérifie le formatage d'une chaîne de résultats."""
    from frontend.helpers import format_results_html

    html = format_results_html("1N2")
    assert "2ecc71" in html  # 1 = vert
    assert "f39c12" in html  # N = orange
    assert "e74c3c" in html  # 2 = rouge
    assert html.count("<span") == 3


def test_format_results_html_empty():
    """Vérifie le formatage d'une chaîne vide."""
    from frontend.helpers import format_results_html

    assert format_results_html("") == ""


def test_result_colors_mapping():
    """Vérifie que les couleurs de résultat sont définies."""
    from frontend.helpers import RESULT_COLORS

    assert "1" in RESULT_COLORS
    assert "N" in RESULT_COLORS
    assert "2" in RESULT_COLORS


def test_grid_type_codes():
    """Vérifie que les codes de grille sont chargés depuis la config."""
    from frontend.helpers import GRID_TYPE_CODES

    assert "LF7" in GRID_TYPE_CODES
    assert GRID_TYPE_CODES["LF7"] == 7


# ---------------------------------------------------------------------------
# Tests match_team_name()
# ---------------------------------------------------------------------------

class _FakeEquipe:
    """Objet minimal simulant une Equipe pour les tests."""
    def __init__(self, nom):
        self.nom = nom


def _make_equipes(*noms):
    return [_FakeEquipe(n) for n in noms]


def test_match_team_name_exact():
    """Match exact (même nom)."""
    from frontend.helpers import match_team_name

    equipes = _make_equipes("Paris Saint Germain", "Olympique de Marseille", "Lyon")
    result = match_team_name("Lyon", equipes)
    assert result is not None
    assert result.nom == "Lyon"


def test_match_team_name_case_insensitive():
    """Match insensible à la casse."""
    from frontend.helpers import match_team_name

    equipes = _make_equipes("Paris Saint Germain", "Lyon")
    result = match_team_name("lyon", equipes)
    assert result is not None
    assert result.nom == "Lyon"


def test_match_team_name_fuzzy_psg():
    """Match fuzzy : 'Paris SG' → 'Paris Saint Germain'."""
    from frontend.helpers import match_team_name

    equipes = _make_equipes("Paris Saint Germain", "Olympique de Marseille", "Lyon")
    result = match_team_name("Paris SG", equipes)
    assert result is not None
    assert result.nom == "Paris Saint Germain"


def test_match_team_name_fuzzy_om():
    """Match fuzzy : 'Marseille' → 'Olympique de Marseille'."""
    from frontend.helpers import match_team_name

    equipes = _make_equipes("Paris Saint Germain", "Olympique de Marseille", "Lyon")
    result = match_team_name("Marseille", equipes)
    assert result is not None
    assert result.nom == "Olympique de Marseille"


def test_match_team_name_accents():
    """Match avec accents : 'Nîmes' vs 'Nimes'."""
    from frontend.helpers import match_team_name

    equipes = _make_equipes("Nimes Olympique")
    result = match_team_name("Nîmes", equipes)
    assert result is not None
    assert result.nom == "Nimes Olympique"


def test_match_team_name_no_false_positive_city():
    """'Orlando City' ne doit PAS matcher 'Man City' (mot commun 'city' trop court)."""
    from frontend.helpers import match_team_name

    equipes = _make_equipes("Man City", "Manchester United", "Liverpool")
    result = match_team_name("Orlando City", equipes)
    assert result is None


def test_match_team_name_no_false_positive_nashville():
    """'Nashville SC' ne doit PAS matcher 'Marseille'."""
    from frontend.helpers import match_team_name

    equipes = _make_equipes("Olympique de Marseille", "Paris Saint Germain", "Lyon")
    result = match_team_name("Nashville SC", equipes)
    assert result is None


def test_match_team_name_no_match():
    """Aucune correspondance suffisante."""
    from frontend.helpers import match_team_name

    equipes = _make_equipes("Paris Saint Germain", "Lyon")
    result = match_team_name("XYZNOTATEAM", equipes)
    assert result is None


def test_match_team_name_empty():
    """Entrées vides."""
    from frontend.helpers import match_team_name

    assert match_team_name("", _make_equipes("Lyon")) is None
    assert match_team_name("Lyon", []) is None


# ---------------------------------------------------------------------------
# Tests fetch_upcoming_grilles() (mock HTTP)
# ---------------------------------------------------------------------------

def test_fetch_upcoming_grilles_format(monkeypatch):
    """Vérifie le format de retour de fetch_upcoming_grilles avec un mock HTML."""
    from collectors import lotofoot_scraper

    sample_html_data = {
        "numero": 5000,
        "type": "LF7",
        "date": None,
        "matchs": [
            {"domicile": "Paris SG", "exterieur": "Marseille", "resultat": None},
            {"domicile": "Lyon", "exterieur": "Monaco", "resultat": None},
        ],
        "rapport_rang1": None,
        "nombre_gagnants": None,
    }

    call_count = {"n": 0}

    def mock_fetch_html(grille_type, grille_id, timeout=15):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return sample_html_data
        return None  # Les IDs suivants n'existent pas

    def mock_get_max_id(grille_type):
        return 4999

    monkeypatch.setattr(lotofoot_scraper, "fetch_grille_html", mock_fetch_html)
    monkeypatch.setattr(lotofoot_scraper, "_get_max_known_id", mock_get_max_id)
    monkeypatch.setattr(lotofoot_scraper.time, "sleep", lambda _: None)

    result = lotofoot_scraper.fetch_upcoming_grilles(
        grille_type="loto-foot-7", max_look_ahead=5, max_consecutive_fails=3
    )

    assert isinstance(result, list)
    assert len(result) == 1

    grille = result[0]
    assert grille["numero"] == 5000
    assert grille["type"] == "LF7"
    assert isinstance(grille["matchs"], list)
    assert len(grille["matchs"]) == 2
    assert "domicile" in grille["matchs"][0]
    assert "exterieur" in grille["matchs"][0]
    # Pas de clé "resultat" dans la sortie (seulement domicile/exterieur)
    assert "resultat" not in grille["matchs"][0]


def test_fetch_upcoming_grilles_skips_played(monkeypatch):
    """Vérifie que les grilles déjà jouées sont ignorées."""
    from collectors import lotofoot_scraper

    played_html_data = {
        "numero": 5000,
        "type": "LF7",
        "date": None,
        "matchs": [
            {"domicile": "Paris SG", "exterieur": "Marseille", "resultat": "1"},
            {"domicile": "Lyon", "exterieur": "Monaco", "resultat": "2"},
        ],
        "rapport_rang1": 1500.0,
        "nombre_gagnants": 3,
    }

    def mock_fetch_html(grille_type, grille_id, timeout=15):
        return played_html_data

    monkeypatch.setattr(lotofoot_scraper, "fetch_grille_html", mock_fetch_html)
    monkeypatch.setattr(lotofoot_scraper, "_get_max_known_id", lambda t: 4999)
    monkeypatch.setattr(lotofoot_scraper.time, "sleep", lambda _: None)

    result = lotofoot_scraper.fetch_upcoming_grilles(
        grille_type="loto-foot-7", max_look_ahead=3
    )

    assert isinstance(result, list)
    assert len(result) == 0
