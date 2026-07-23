"""Tests pour le scraper Pronosoft."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collectors.pronosoft_scraper import (
    parse_pronosoft_html,
    _parse_grille_header,
    parse_combinaisons_html,
    fetch_combinaisons_stats,
    _combinaisons_cache,
)


# === HTML mocké reproduisant la structure réelle de Pronosoft ===

SAMPLE_HTML = """
<html><body>
<div class="clearer">
<table class="grilles-lf">
    <tbody>
        <tr><th colspan="4">LF 15 n&deg;57</th></tr>
        <tr class="valid"><th colspan="4">Avant le
            <span data-date-utc="2026-07-23 16:55:00" data-date-format="text_full">
                Jeudi 23 Juillet à 18h55
            </span>
        </th></tr>
        <tr>
            <td>1</td>
            <td class="equipe home">Saint Gall</td>
            <td class="sep">-</td>
            <td class="equipe ext">Benfica</td>
        </tr>
        <tr>
            <td>2</td>
            <td class="equipe home">Besiktas</td>
            <td class="sep">-</td>
            <td class="equipe ext">Midtjylland</td>
        </tr>
        <tr>
            <td>3</td>
            <td class="equipe home">Twente</td>
            <td class="sep">-</td>
            <td class="equipe ext">Ferencvaros</td>
        </tr>
        <tr>
            <td>4</td>
            <td class="equipe home">Hajduk Split</td>
            <td class="sep">-</td>
            <td class="equipe ext">Pafos FC</td>
        </tr>
        <tr>
            <td>5</td>
            <td class="equipe home">Hammarby</td>
            <td class="sep">-</td>
            <td class="equipe ext">Anderlecht</td>
        </tr>
        <tr>
            <td>6</td>
            <td class="equipe home">Tromso</td>
            <td class="sep">-</td>
            <td class="equipe ext">Hradec Kralove</td>
        </tr>
        <tr>
            <td>7</td>
            <td class="equipe home">Sheriff Tiraspol</td>
            <td class="sep">-</td>
            <td class="equipe ext">Mac. Tel Aviv</td>
        </tr>
        <tr>
            <td>8</td>
            <td class="equipe home">Dynamo Kiev</td>
            <td class="sep">-</td>
            <td class="equipe ext">PAOK Salonique</td>
        </tr>
        <tr>
            <td>9</td>
            <td class="equipe home">HNK Rijeka</td>
            <td class="sep">-</td>
            <td class="equipe ext">Derry City</td>
        </tr>
        <tr>
            <td>10</td>
            <td class="equipe home">Vojvodina</td>
            <td class="sep">-</td>
            <td class="equipe ext">Ajax</td>
        </tr>
        <tr>
            <td>11</td>
            <td class="equipe home">Paksi SE</td>
            <td class="sep">-</td>
            <td class="equipe ext">Panathinaikos</td>
        </tr>
        <tr>
            <td>12</td>
            <td class="equipe home">Zhytomyr</td>
            <td class="sep">-</td>
            <td class="equipe ext">FC Copenhague</td>
        </tr>
        <tr>
            <td>13</td>
            <td class="equipe home">LNZ Cherkasy</td>
            <td class="sep">-</td>
            <td class="equipe ext">La Gantoise</td>
        </tr>
        <tr>
            <td>14</td>
            <td class="equipe home">Universit Cluj</td>
            <td class="sep">-</td>
            <td class="equipe ext">SK Brann</td>
        </tr>
        <tr>
            <td>15</td>
            <td class="equipe home">GAIS Goteborg</td>
            <td class="sep">-</td>
            <td class="equipe ext">Nordsjaelland</td>
        </tr>
    </tbody>
</table>
<table class="grilles-lf">
    <tbody>
        <tr><th colspan="4">LF 8 n&deg;95</th></tr>
        <tr class="valid"><th colspan="4">Avant le
            <span data-date-utc="2026-07-23 16:55:00" data-date-format="text_full">
                Jeudi 23 Juillet à 18h55
            </span>
        </th></tr>
        <tr>
            <td>1</td>
            <td class="equipe home">Saint Gall</td>
            <td class="sep">-</td>
            <td class="equipe ext">Benfica</td>
        </tr>
        <tr>
            <td>2</td>
            <td class="equipe home">Besiktas</td>
            <td class="sep">-</td>
            <td class="equipe ext">Midtjylland</td>
        </tr>
        <tr>
            <td>3</td>
            <td class="equipe home">Twente</td>
            <td class="sep">-</td>
            <td class="equipe ext">Ferencvaros</td>
        </tr>
        <tr>
            <td>4</td>
            <td class="equipe home">Hajduk Split</td>
            <td class="sep">-</td>
            <td class="equipe ext">Pafos FC</td>
        </tr>
        <tr>
            <td>5</td>
            <td class="equipe home">Hammarby</td>
            <td class="sep">-</td>
            <td class="equipe ext">Anderlecht</td>
        </tr>
        <tr>
            <td>6</td>
            <td class="equipe home">Tromso</td>
            <td class="sep">-</td>
            <td class="equipe ext">Hradec Kralove</td>
        </tr>
        <tr>
            <td>7</td>
            <td class="equipe home">Sheriff Tiraspol</td>
            <td class="sep">-</td>
            <td class="equipe ext">Mac. Tel Aviv</td>
        </tr>
        <tr>
            <td>8</td>
            <td class="equipe home">Dynamo Kiev</td>
            <td class="sep">-</td>
            <td class="equipe ext">PAOK Salonique</td>
        </tr>
    </tbody>
</table>
<table class="grilles-lf">
    <tbody>
        <tr><th colspan="4">LF 7 n&deg;92</th></tr>
        <tr class="valid"><th colspan="4">Avant le
            <span data-date-utc="2026-07-24 16:55:00" data-date-format="text_full">
                Vendredi 24 Juillet à 18h55
            </span>
        </th></tr>
        <tr>
            <td>1</td>
            <td class="equipe home">St Patricks</td>
            <td class="sep">-</td>
            <td class="equipe ext">Dundalk</td>
        </tr>
        <tr>
            <td>2</td>
            <td class="equipe home">FC Arges</td>
            <td class="sep">-</td>
            <td class="equipe ext">Pet. Ploiesti</td>
        </tr>
        <tr>
            <td>3</td>
            <td class="equipe home">Pogon Szczecin</td>
            <td class="sep">-</td>
            <td class="equipe ext">Legia Varsovie</td>
        </tr>
        <tr>
            <td>4</td>
            <td class="equipe home">FC Arda</td>
            <td class="sep">-</td>
            <td class="equipe ext">Slavia Sofia</td>
        </tr>
        <tr>
            <td>5</td>
            <td class="equipe home">Elfsborg</td>
            <td class="sep">-</td>
            <td class="equipe ext">Hacken</td>
        </tr>
        <tr>
            <td>6</td>
            <td class="equipe home">Drogheda Utd</td>
            <td class="sep">-</td>
            <td class="equipe ext">Bohemians</td>
        </tr>
        <tr>
            <td>7</td>
            <td class="equipe home">Viborg</td>
            <td class="sep">-</td>
            <td class="equipe ext">Aalborg</td>
        </tr>
    </tbody>
</table>
<table class="grilles-lf">
    <tbody>
        <tr><th colspan="4">LF 7 - LF 12</th></tr>
        <tr class="valid"><th colspan="4">Avant le
            <span data-date-utc="2026-08-02 14:25:00" data-date-format="text_full">
                Dimanche 2 Août à 16h25
            </span>
        </th></tr>
        <tr>
            <td>1</td>
            <td class="equipe home">Grasshoppers</td>
            <td class="sep">-</td>
            <td class="equipe ext">FC Lugano</td>
        </tr>
        <tr>
            <td>2</td>
            <td class="equipe home">Sion</td>
            <td class="sep">-</td>
            <td class="equipe ext">Lucerne</td>
        </tr>
        <tr>
            <td>3</td>
            <td class="equipe home">Rapid Bucarest</td>
            <td class="sep">-</td>
            <td class="equipe ext">Cfr Cluj</td>
        </tr>
        <tr>
            <td>4</td>
            <td class="equipe home">Silkeborg</td>
            <td class="sep">-</td>
            <td class="equipe ext">FC Copenhague</td>
        </tr>
        <tr>
            <td>5</td>
            <td class="equipe home">Dynamo Kiev</td>
            <td class="sep">-</td>
            <td class="equipe ext">Livyi Bereg</td>
        </tr>
        <tr>
            <td>6</td>
            <td class="equipe home">Hibernian</td>
            <td class="sep">-</td>
            <td class="equipe ext">Motherwell</td>
        </tr>
        <tr>
            <td>7</td>
            <td class="equipe home">SK Brann</td>
            <td class="sep">-</td>
            <td class="equipe ext">Rosenborg</td>
        </tr>
        <tr>
            <td>8</td>
            <td class="equipe home">Aalesunds</td>
            <td class="sep">-</td>
            <td class="equipe ext">Tromso</td>
        </tr>
        <tr>
            <td>9</td>
            <td class="equipe home">Rapid Vienne</td>
            <td class="sep">-</td>
            <td class="equipe ext">SCR Altach</td>
        </tr>
        <tr>
            <td>10</td>
            <td class="equipe home">Wolfsberger AC</td>
            <td class="sep">-</td>
            <td class="equipe ext">Austria Vienne</td>
        </tr>
        <tr>
            <td>11</td>
            <td class="equipe home">Hunedoara</td>
            <td class="sep">-</td>
            <td class="equipe ext">ACS Sepsi</td>
        </tr>
        <tr>
            <td>12</td>
            <td class="equipe home">AIK Solna</td>
            <td class="sep">-</td>
            <td class="equipe ext">Orgryte</td>
        </tr>
    </tbody>
</table>
</div>
</body></html>
"""


# === Tests parsing header ===


class TestParseGrilleHeader:
    def test_type_with_numero(self):
        result = _parse_grille_header("LF 15 n°57")
        assert len(result) == 1
        assert result[0]["type"] == "LF15"
        assert result[0]["numero"] == 57

    def test_type_without_numero(self):
        result = _parse_grille_header("LF 15")
        assert len(result) == 1
        assert result[0]["type"] == "LF15"
        assert result[0]["numero"] is None

    def test_combined_types(self):
        result = _parse_grille_header("LF 7 - LF 12")
        assert len(result) == 2
        types = {r["type"] for r in result}
        assert types == {"LF7", "LF12"}
        assert all(r["numero"] is None for r in result)

    def test_lf7_with_numero(self):
        result = _parse_grille_header("LF 7 n°92")
        assert len(result) == 1
        assert result[0]["type"] == "LF7"
        assert result[0]["numero"] == 92

    def test_lf8_with_numero(self):
        result = _parse_grille_header("LF 8 n°95")
        assert len(result) == 1
        assert result[0]["type"] == "LF8"
        assert result[0]["numero"] == 95


# === Tests parsing HTML complet ===


class TestParsePronosoftHtml:
    def test_returns_list(self):
        result = parse_pronosoft_html(SAMPLE_HTML)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_types_present(self):
        result = parse_pronosoft_html(SAMPLE_HTML)
        types = {g["type"] for g in result}
        assert "LF15" in types
        assert "LF8" in types
        assert "LF7" in types

    def test_lf15_grille(self):
        result = parse_pronosoft_html(SAMPLE_HTML)
        lf15 = [g for g in result if g["type"] == "LF15"]
        assert len(lf15) >= 1
        g = lf15[0]
        assert g["numero"] == 57
        assert g["date"] is not None
        assert len(g["matchs"]) == 15

    def test_lf8_grille(self):
        result = parse_pronosoft_html(SAMPLE_HTML)
        lf8 = [g for g in result if g["type"] == "LF8"]
        assert len(lf8) >= 1
        g = lf8[0]
        assert g["numero"] == 95
        assert len(g["matchs"]) == 8

    def test_lf7_grille(self):
        result = parse_pronosoft_html(SAMPLE_HTML)
        lf7 = [g for g in result if g["type"] == "LF7"]
        assert len(lf7) >= 1
        # La première LF7 a un numéro, la seconde (combinée LF7-LF12) n'en a pas
        numbered = [g for g in lf7 if g["numero"] is not None]
        assert len(numbered) >= 1
        assert numbered[0]["numero"] == 92
        assert len(numbered[0]["matchs"]) == 7

    def test_combined_lf7_lf12(self):
        """Test que la table combinée LF 7 - LF 12 produit 2 grilles."""
        result = parse_pronosoft_html(SAMPLE_HTML)
        lf12 = [g for g in result if g["type"] == "LF12"]
        assert len(lf12) >= 1
        assert len(lf12[0]["matchs"]) == 12

    def test_match_format(self):
        """Chaque match a les clés domicile et exterieur."""
        result = parse_pronosoft_html(SAMPLE_HTML)
        for grille in result:
            for match in grille["matchs"]:
                assert "domicile" in match
                assert "exterieur" in match
                assert isinstance(match["domicile"], str)
                assert isinstance(match["exterieur"], str)
                assert len(match["domicile"]) > 0
                assert len(match["exterieur"]) > 0

    def test_date_parsed(self):
        """Les dates sont correctement parsées depuis data-date-utc."""
        result = parse_pronosoft_html(SAMPLE_HTML)
        from datetime import date
        lf15 = [g for g in result if g["type"] == "LF15"][0]
        assert lf15["date"] == date(2026, 7, 23)

    def test_first_match_teams(self):
        """Vérifie les noms d'équipes du premier match LF15."""
        result = parse_pronosoft_html(SAMPLE_HTML)
        lf15 = [g for g in result if g["type"] == "LF15"][0]
        assert lf15["matchs"][0]["domicile"] == "Saint Gall"
        assert lf15["matchs"][0]["exterieur"] == "Benfica"


# === Tests filtrage par type ===


class TestFilterByType:
    def test_filter_lf7(self):
        all_grilles = parse_pronosoft_html(SAMPLE_HTML)
        lf7 = [g for g in all_grilles if g["type"] == "LF7"]
        assert all(g["type"] == "LF7" for g in lf7)
        assert len(lf7) >= 1

    def test_filter_lf15(self):
        all_grilles = parse_pronosoft_html(SAMPLE_HTML)
        lf15 = [g for g in all_grilles if g["type"] == "LF15"]
        assert all(g["type"] == "LF15" for g in lf15)

    def test_filter_nonexistent(self):
        all_grilles = parse_pronosoft_html(SAMPLE_HTML)
        unknown = [g for g in all_grilles if g["type"] == "LF99"]
        assert unknown == []


# === Test HTML vide ===


class TestEdgeCases:
    def test_empty_html(self):
        result = parse_pronosoft_html("")
        assert result == []

    def test_no_tables(self):
        result = parse_pronosoft_html("<html><body><p>Rien</p></body></html>")
        assert result == []

    def test_table_without_matches(self):
        html = """
        <table class="grilles-lf">
            <tbody>
                <tr><th colspan="4">LF 7 n°99</th></tr>
                <tr class="valid"><th colspan="4">Avant le
                    <span data-date-utc="2026-01-01 12:00:00">Jeudi 1 Janvier</span>
                </th></tr>
            </tbody>
        </table>
        """
        result = parse_pronosoft_html(html)
        assert result == []


# === HTML mocké pour la table combinaisons 1N2 ===

SAMPLE_COMBINAISONS_HTML = """
<html><body>
<table class="stat_1n2">
    <tr><th colspan="4">Combinaisons 1N2</th></tr>
    <tr><th>1</th><th>N</th><th>2</th><th>Sorties</th></tr>
    <tr><td>3</td><td>2</td><td>2</td><td>532 fois</td></tr>
    <tr><td>4</td><td>1</td><td>2</td><td>459 fois</td></tr>
    <tr><td>4</td><td>2</td><td>1</td><td>480 fois</td></tr>
    <tr><td>5</td><td>1</td><td>1</td><td>350 fois</td></tr>
    <tr><td>3</td><td>3</td><td>1</td><td>310 fois</td></tr>
    <tr><td>2</td><td>3</td><td>2</td><td>290 fois</td></tr>
    <tr><td>7</td><td>0</td><td>0</td><td>23 fois</td></tr>
</table>
<!-- Autre table stat_1n2 qui ne contient pas "Combinaisons 1N2" -->
<table class="stat_1n2">
    <tr><th colspan="4">Signes 1</th></tr>
    <tr><td>99</td><td>0</td><td>0</td><td>1 fois</td></tr>
</table>
</body></html>
"""


# === Tests parsing combinaisons 1N2 ===


class TestParseCombinaisons:
    def test_returns_dict(self):
        result = parse_combinaisons_html(SAMPLE_COMBINAISONS_HTML)
        assert isinstance(result, dict)

    def test_correct_profiles_count(self):
        result = parse_combinaisons_html(SAMPLE_COMBINAISONS_HTML)
        assert len(result) == 7

    def test_profile_values(self):
        result = parse_combinaisons_html(SAMPLE_COMBINAISONS_HTML)
        assert result["3-2-2"] == 532
        assert result["4-1-2"] == 459
        assert result["7-0-0"] == 23

    def test_ignores_other_tables(self):
        """Ne doit pas parser la table 'Signes 1'."""
        result = parse_combinaisons_html(SAMPLE_COMBINAISONS_HTML)
        assert "99-0-0" not in result

    def test_empty_html(self):
        result = parse_combinaisons_html("")
        assert result == {}

    def test_no_matching_table(self):
        html = '<table class="stat_1n2"><tr><th>Autre</th></tr></table>'
        result = parse_combinaisons_html(html)
        assert result == {}


class TestNormalisationCombinaisons:
    def test_proportions_sum_to_one(self):
        counts = parse_combinaisons_html(SAMPLE_COMBINAISONS_HTML)
        total = sum(counts.values())
        proportions = {k: v / total for k, v in counts.items()}
        assert abs(sum(proportions.values()) - 1.0) < 1e-10

    def test_most_frequent_profile(self):
        counts = parse_combinaisons_html(SAMPLE_COMBINAISONS_HTML)
        total = sum(counts.values())
        proportions = {k: v / total for k, v in counts.items()}
        most_frequent = max(proportions, key=proportions.get)
        assert most_frequent == "3-2-2"

    def test_rare_profile_has_low_proportion(self):
        counts = parse_combinaisons_html(SAMPLE_COMBINAISONS_HTML)
        total = sum(counts.values())
        proportions = {k: v / total for k, v in counts.items()}
        assert proportions["7-0-0"] < 0.02


class TestCombinaionsCache:
    def test_cache_prevents_refetch(self, monkeypatch):
        """Le cache doit empêcher un second appel réseau."""
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            class MockResp:
                status_code = 200
                text = SAMPLE_COMBINAISONS_HTML
                def raise_for_status(self):
                    pass

            return MockResp()

        # Nettoyer le cache avant le test
        _combinaisons_cache.clear()

        monkeypatch.setattr("collectors.pronosoft_scraper.requests.get", mock_get)
        result1 = fetch_combinaisons_stats("LF7")
        result2 = fetch_combinaisons_stats("LF7")

        assert call_count == 1
        assert result1 == result2
        assert len(result1) == 7

        # Nettoyer après
        _combinaisons_cache.clear()

    def test_unknown_grid_type(self):
        result = fetch_combinaisons_stats("LF99")
        assert result == {}
