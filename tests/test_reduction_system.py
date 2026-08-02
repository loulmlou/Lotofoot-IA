"""Tests pour le module reduction_system — systèmes réducteurs à garanties."""

import pytest

from generator.reduction_system import build_reduction_system


# =====================================================
# Fixtures
# =====================================================

@pytest.fixture
def selections_simple():
    """3 matchs bancos, 2 doubles : 1 * 1 * 1 * 2 * 2 = 4 combinaisons."""
    return [["1"], ["N"], ["2"], ["1", "N"], ["1", "2"]]


@pytest.fixture
def selections_triples():
    """3 triples : 3 * 3 * 3 = 27 combinaisons."""
    return [["1", "N", "2"], ["1", "N", "2"], ["1", "N", "2"]]


# =====================================================
# Validation des entrées
# =====================================================

class TestValidation:
    def test_empty_selections_raises(self):
        with pytest.raises(ValueError):
            build_reduction_system([], [(1, 1.0)])

    def test_empty_match_options_raises(self):
        with pytest.raises(ValueError):
            build_reduction_system([[]], [(0, 1.0)])

    def test_invalid_result_raises(self):
        with pytest.raises(ValueError):
            build_reduction_system([["1", "X"]], [(0, 1.0)])

    def test_duplicate_result_raises(self):
        with pytest.raises(ValueError):
            build_reduction_system([["1", "1"]], [(0, 1.0)])

    def test_empty_guarantees_raises(self, selections_simple):
        with pytest.raises(ValueError):
            build_reduction_system(selections_simple, [])

    def test_radius_too_large_raises(self, selections_simple):
        with pytest.raises(ValueError):
            build_reduction_system(selections_simple, [(len(selections_simple), 1.0)])

    def test_negative_radius_raises(self, selections_simple):
        with pytest.raises(ValueError):
            build_reduction_system(selections_simple, [(-1, 1.0)])

    def test_coverage_out_of_range_raises(self, selections_simple):
        with pytest.raises(ValueError):
            build_reduction_system(selections_simple, [(1, 1.5)])
        with pytest.raises(ValueError):
            build_reduction_system(selections_simple, [(1, 0.0)])

    def test_too_many_combinations_raises(self):
        # 9 matchs en triple = 3**9 = 19683 > MAX_COMBINATIONS (5000)
        selections = [["1", "N", "2"]] * 9
        with pytest.raises(ValueError):
            build_reduction_system(selections, [(1, 1.0)])


# =====================================================
# Garantie 100% (full covering) == multiple complète
# =====================================================

class TestFullGuarantee:
    def test_radius_0_full_coverage_needs_all_combos(self, selections_simple):
        """Garantie n-0 (tout juste) à 100% => il faut jouer toutes les combos."""
        result = build_reduction_system(selections_simple, [(0, 1.0)])
        assert result["nb_grilles"] == result["nb_combinaisons_total"] == 4
        assert result["couverture"][0] == 1.0
        assert result["taux_reduction"] == 0.0

    def test_all_grilles_within_selections(self, selections_triples):
        result = build_reduction_system(selections_triples, [(0, 1.0)])
        for g in result["grilles"]:
            for i, c in enumerate(g["resultats"]):
                assert c in selections_triples[i]

    def test_grilles_are_unique(self, selections_triples):
        result = build_reduction_system(selections_triples, [(1, 1.0)])
        resultats = [g["resultats"] for g in result["grilles"]]
        assert len(resultats) == len(set(resultats))


# =====================================================
# Réduction avec garantie partielle
# =====================================================

class TestReduction:
    def test_reduction_uses_fewer_grids_than_full(self, selections_triples):
        """n-1 à 100% doit nécessiter moins de grilles que la multiple complète (27)."""
        result = build_reduction_system(selections_triples, [(1, 1.0)])
        assert result["nb_grilles"] < result["nb_combinaisons_total"]
        assert result["couverture"][1] == 1.0

    def test_partial_guarantee_uses_even_fewer_grids(self, selections_triples):
        strict = build_reduction_system(selections_triples, [(1, 1.0)])
        partiel = build_reduction_system(selections_triples, [(1, 0.5)])
        assert partiel["nb_grilles"] <= strict["nb_grilles"]
        assert partiel["couverture"][1] >= 0.5

    def test_multi_tier_guarantees(self, selections_triples):
        """n-1 100% et n-2 75% simultanément."""
        result = build_reduction_system(selections_triples, [(1, 1.0), (2, 0.75)])
        assert result["couverture"][1] >= 1.0 - 1e-9
        assert result["couverture"][2] >= 0.75 - 1e-9
        # Le rayon le plus large doit couvrir au moins autant que le plus étroit
        assert result["couverture"][2] >= result["couverture"][1]

    def test_single_combination_shortcut(self):
        result = build_reduction_system([["1"], ["N"], ["2"]], [(0, 1.0)])
        assert result["nb_grilles"] == 1
        assert result["grilles"][0]["resultats"] == "1N2"
        assert result["taux_reduction"] == 0.0


# =====================================================
# Structure du résultat
# =====================================================

class TestResultStructure:
    def test_keys_present(self, selections_simple):
        result = build_reduction_system(selections_simple, [(1, 1.0)])
        for key in ("grilles", "nb_grilles", "nb_combinaisons_total",
                    "taux_reduction", "couverture"):
            assert key in result

    def test_grille_length_matches_n_matchs(self, selections_simple):
        result = build_reduction_system(selections_simple, [(0, 1.0)])
        for g in result["grilles"]:
            assert len(g["resultats"]) == len(selections_simple)

    def test_nb_grilles_matches_list_length(self, selections_triples):
        result = build_reduction_system(selections_triples, [(1, 0.8)])
        assert result["nb_grilles"] == len(result["grilles"])
