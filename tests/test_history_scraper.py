"""Tests pour le scraper historique Pronosoft (collectors/pronosoft_history.py)."""

import json
import os
import tempfile

import pytest

from collectors.pronosoft_history import (
    parse_repartition_html,
    parse_historique_html,
    compute_grid_metrics,
    save_history,
    load_history,
)


# ---------------------------------------------------------------------------
# Fixtures HTML mockées
# ---------------------------------------------------------------------------

REPARTITION_HTML = """
<html><body>
<h3>Indice de difficulté</h3>
<p>7,14</p>
<p>Répartition basée sur 821 pronostics</p>

<table class="prono-cyb-des">
<tr>
  <td class="match">Inter Miami - Chicago Fire</td>
  <td class="cote-bet">64.9%1.35</td>
  <td class="cote-d">20.4%3.80</td>
  <td class="cote-d">14.7%5.50</td>
  <td class="prono_match">1</td>
  <td class="dev_desktop_td_score">3-2</td>
</tr>
<tr>
  <td class="match">NY Red Bulls - Columbus Crew</td>
  <td class="cote-d">35.2%2.25</td>
  <td class="cote-d">28.1%3.40</td>
  <td class="cote-bet">36.7%2.50</td>
  <td class="prono_match">N2</td>
  <td class="dev_desktop_td_score">1-1</td>
</tr>
<tr>
  <td class="match">FC Dallas - Houston Dynamo</td>
  <td class="cote-d">40.0%1.85</td>
  <td class="cote-d">30.0%3.50</td>
  <td class="cote-d">30.0%3.80</td>
  <td class="prono_match">1</td>
  <td class="dev_desktop_td_score">2-0</td>
</tr>
</table>
</body></html>
"""

HISTORIQUE_HTML = """
<html><body>
<table class="hist">
<tr>
  <td>1</td>
  <td>Inter Miami - Chicago Fire</td>
  <td><span class="res">1</span></td>
</tr>
<tr>
  <td>2</td>
  <td>NY Red Bulls - Columbus Crew</td>
  <td><span class="res">N</span></td>
</tr>
<tr>
  <td>3</td>
  <td>FC Dallas - Houston Dynamo</td>
  <td><span class="res">1</span></td>
</tr>
</table>

<table>
<tr><th>Rapport des gagnants</th></tr>
<tr>
  <td>3 sur 3</td>
  <td>22</td>
  <td>1 210,00 €</td>
</tr>
<tr>
  <td>2 sur 3</td>
  <td>353</td>
  <td>92,10 €</td>
</tr>
</table>
</body></html>
"""


# ---------------------------------------------------------------------------
# Tests parse_repartition_html
# ---------------------------------------------------------------------------

class TestParseRepartition:
    def test_parse_matches(self):
        result = parse_repartition_html(REPARTITION_HTML)
        assert len(result["matches"]) == 3

    def test_parse_teams(self):
        result = parse_repartition_html(REPARTITION_HTML)
        m1 = result["matches"][0]
        assert m1["home"] == "Inter Miami"
        assert m1["away"] == "Chicago Fire"

    def test_parse_cotes(self):
        result = parse_repartition_html(REPARTITION_HTML)
        m1 = result["matches"][0]
        assert m1["cote_1"] == 1.35
        assert m1["cote_n"] == 3.80
        assert m1["cote_2"] == 5.50

    def test_parse_percentages(self):
        result = parse_repartition_html(REPARTITION_HTML)
        m1 = result["matches"][0]
        assert m1["pct_1"] == 64.9
        assert m1["pct_n"] == 20.4
        assert m1["pct_2"] == 14.7

    def test_parse_prono_cyborg(self):
        result = parse_repartition_html(REPARTITION_HTML)
        assert result["matches"][0]["prono_cyborg"] == "1"
        assert result["matches"][1]["prono_cyborg"] == "N2"

    def test_parse_score(self):
        result = parse_repartition_html(REPARTITION_HTML)
        assert result["matches"][0]["score"] == "3-2"

    def test_parse_difficulty(self):
        result = parse_repartition_html(REPARTITION_HTML)
        assert result["difficulty"] == 7.14

    def test_parse_nb_pronostics(self):
        result = parse_repartition_html(REPARTITION_HTML)
        assert result["nb_pronostics"] == 821

    def test_parse_position(self):
        result = parse_repartition_html(REPARTITION_HTML)
        assert result["matches"][0]["position"] == 1
        assert result["matches"][2]["position"] == 3

    def test_empty_html(self):
        result = parse_repartition_html("<html><body></body></html>")
        assert result["matches"] == []
        assert result["difficulty"] is None
        assert result["nb_pronostics"] is None


# ---------------------------------------------------------------------------
# Tests parse_historique_html
# ---------------------------------------------------------------------------

class TestParseHistorique:
    def test_parse_resultats(self):
        result = parse_historique_html(HISTORIQUE_HTML)
        assert result["resultats_par_match"] == ["1", "N", "1"]
        assert result["resultats_str"] == "1N1"

    def test_parse_profil(self):
        result = parse_historique_html(HISTORIQUE_HTML)
        assert result["profil"] == "2-1-0"

    def test_parse_rapports(self):
        result = parse_historique_html(HISTORIQUE_HTML)
        assert "3_sur_3" in result["rapports"]
        r = result["rapports"]["3_sur_3"]
        assert r["gagnants"] == 22
        assert r["montant"] == 1210.0

    def test_parse_rapports_rang2(self):
        result = parse_historique_html(HISTORIQUE_HTML)
        assert "2_sur_3" in result["rapports"]
        r = result["rapports"]["2_sur_3"]
        assert r["gagnants"] == 353
        assert r["montant"] == 92.1

    def test_empty_html(self):
        result = parse_historique_html("<html><body></body></html>")
        assert result["resultats_par_match"] == []
        assert result["resultats_str"] == ""
        assert result["profil"] == ""


# ---------------------------------------------------------------------------
# Tests compute_grid_metrics
# ---------------------------------------------------------------------------

class TestComputeGridMetrics:
    def test_basic_metrics(self):
        grid = {
            "matches": [
                {"cote_1": 1.35, "cote_n": 3.80, "cote_2": 5.50, "resultat": "1"},
                {"cote_1": 2.25, "cote_n": 3.40, "cote_2": 2.50, "resultat": "N"},
                {"cote_1": 1.85, "cote_n": 3.50, "cote_2": 3.80, "resultat": "1"},
            ]
        }
        metrics = compute_grid_metrics(grid)
        # Favorites: 1.35, 2.25, 1.85
        assert metrics["somme_cotes_fav"] == round(1.35 + 2.25 + 1.85, 2)
        assert metrics["moyenne_cote_fav"] == round((1.35 + 2.25 + 1.85) / 3, 2)
        assert metrics["nb_surprises"] == 1  # Match 2: fav=1 (2.25), result=N

    def test_empty_matches(self):
        metrics = compute_grid_metrics({"matches": []})
        assert metrics["somme_cotes_fav"] == 0.0
        assert metrics["nb_surprises"] == 0

    def test_no_surprises(self):
        grid = {
            "matches": [
                {"cote_1": 1.20, "cote_n": 5.0, "cote_2": 8.0, "resultat": "1"},
                {"cote_1": 1.30, "cote_n": 4.5, "cote_2": 7.0, "resultat": "1"},
            ]
        }
        metrics = compute_grid_metrics(grid)
        assert metrics["nb_surprises"] == 0

    def test_all_surprises(self):
        grid = {
            "matches": [
                {"cote_1": 1.20, "cote_n": 5.0, "cote_2": 8.0, "resultat": "2"},
                {"cote_1": 1.30, "cote_n": 4.5, "cote_2": 7.0, "resultat": "N"},
            ]
        }
        metrics = compute_grid_metrics(grid)
        assert metrics["nb_surprises"] == 2

    def test_ecart_type(self):
        grid = {
            "matches": [
                {"cote_1": 1.50, "cote_n": 3.0, "cote_2": 5.0},
                {"cote_1": 1.50, "cote_n": 3.0, "cote_2": 5.0},
            ]
        }
        metrics = compute_grid_metrics(grid)
        assert metrics["ecart_type_cotes"] == 0.0  # All same


# ---------------------------------------------------------------------------
# Tests save/load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self):
        grids = [
            {
                "grid_number": 1,
                "matches": [
                    {"home": "A", "away": "B", "cote_1": 1.5, "cote_n": 3.5, "cote_2": 4.0},
                ],
                "difficulty": 5.0,
                "resultats": "1",
                "profil": "1-0-0",
            },
            {
                "grid_number": 2,
                "matches": [
                    {"home": "C", "away": "D", "cote_1": 2.0, "cote_n": 3.0, "cote_2": 3.5},
                ],
                "difficulty": 6.5,
                "resultats": "N",
                "profil": "0-1-0",
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
        ) as f:
            path = f.name

        try:
            save_history(grids, path)
            loaded = load_history(path)
            assert len(loaded) == 2
            assert loaded[0]["grid_number"] == 1
            assert loaded[1]["grid_number"] == 2
            assert loaded[0]["matches"][0]["home"] == "A"
            assert loaded[0]["difficulty"] == 5.0
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        result = load_history("/nonexistent/path.json")
        assert result == []
