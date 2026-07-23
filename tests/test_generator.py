"""Tests pour le module generator (Phase 4) — génération de grilles optimisées."""

import pytest

from models.predictor import Predictor
from generator.grid_generator import GridGenerator
from generator.optimizer import optimize_grids, compute_grid_probability, compute_grid_expected_value, _compute_grid_profile


# =====================================================
# Fixture : features synthétiques pour les tests
# =====================================================

@pytest.fixture
def sample_features_list():
    """Retourne 7 dicts de features simulant une grille LF7."""
    base = {
        "dom_forme_pts": 10, "dom_forme_ratio_vic": 0.6,
        "dom_forme_buts_m": 1.8, "dom_forme_buts_e": 0.8, "dom_forme_nb": 5,
        "ext_forme_pts": 7, "ext_forme_ratio_vic": 0.4,
        "ext_forme_buts_m": 1.2, "ext_forme_buts_e": 1.4, "ext_forme_nb": 5,
        "h2h_nb": 3, "h2h_pct_dom": 0.5, "h2h_pct_nul": 0.2, "h2h_pct_ext": 0.3,
        "dom_position": 3.0, "ext_position": 10.0, "diff_position": -7.0,
        "dom_classement_pts": 45.0, "ext_classement_pts": 28.0,
        "dom_moy_buts_m": 1.9, "dom_moy_buts_e": 0.7, "dom_over25": 0.6,
        "ext_moy_buts_m": 1.3, "ext_moy_buts_e": 1.5, "ext_over25": 0.5,
        "prob_1": 0.55, "prob_n": 0.25, "prob_2": 0.20,
        "cote_surprise": 0.55, "cote_1": 1.8, "cote_n": 3.5, "cote_2": 4.5,
        "dom_jours_repos": 7.0, "ext_jours_repos": 4.0,
        "avantage_dom_ligue": 0.46,
    }

    matches = []
    # Varier les features pour avoir des confiances différentes
    configs = [
        {"prob_1": 0.60, "prob_n": 0.22, "prob_2": 0.18},  # Fort favori dom
        {"prob_1": 0.35, "prob_n": 0.30, "prob_2": 0.35},  # Match équilibré
        {"prob_1": 0.20, "prob_n": 0.25, "prob_2": 0.55},  # Fort favori ext
        {"prob_1": 0.45, "prob_n": 0.28, "prob_2": 0.27},  # Léger favori dom
        {"prob_1": 0.50, "prob_n": 0.25, "prob_2": 0.25},  # Favori dom moyen
        {"prob_1": 0.33, "prob_n": 0.34, "prob_2": 0.33},  # Très équilibré
        {"prob_1": 0.42, "prob_n": 0.30, "prob_2": 0.28},  # Léger favori dom
    ]

    for cfg in configs:
        m = dict(base)
        m.update(cfg)
        # Ajuster les cotes en cohérence
        m["cote_1"] = 1.0 / cfg["prob_1"] if cfg["prob_1"] > 0 else 10.0
        m["cote_n"] = 1.0 / cfg["prob_n"] if cfg["prob_n"] > 0 else 10.0
        m["cote_2"] = 1.0 / cfg["prob_2"] if cfg["prob_2"] > 0 else 10.0
        matches.append(m)

    return matches


@pytest.fixture
def predictor():
    """Predictor sans modèle ML (scoring pondéré seul)."""
    return Predictor(model_path="/nonexistent/path.joblib", strategy="equilibree")


@pytest.fixture
def generator(predictor):
    """GridGenerator avec predictor de test."""
    return GridGenerator(predictor=predictor, strategy="equilibree")


# =====================================================
# Tests de GridGenerator.generate()
# =====================================================

class TestGridGeneratorGenerate:
    def test_returns_list(self, generator, sample_features_list):
        grids = generator.generate(sample_features_list, grid_type="LF7", budget=5)
        assert isinstance(grids, list)

    def test_returns_correct_count(self, generator, sample_features_list):
        grids = generator.generate(sample_features_list, grid_type="LF7", budget=5)
        assert len(grids) <= 5
        assert len(grids) >= 1

    def test_grid_structure(self, generator, sample_features_list):
        grids = generator.generate(sample_features_list, grid_type="LF7", budget=3)
        for g in grids:
            assert "resultats" in g
            assert "confiance" in g
            assert "probabilite" in g
            assert "matchs" in g
            assert len(g["resultats"]) == 7
            assert all(c in "1N2" for c in g["resultats"])

    def test_budget_respected(self, generator, sample_features_list):
        for budget in [1, 3, 5, 10]:
            grids = generator.generate(sample_features_list, grid_type="LF7", budget=budget)
            assert len(grids) <= budget

    def test_empty_matches(self, generator):
        grids = generator.generate([], grid_type="LF7", budget=5)
        assert grids == []


# =====================================================
# Tests de la grille de base
# =====================================================

class TestBaseGrid:
    def test_base_grid_is_favorites(self, generator, sample_features_list):
        predictions = generator._predict_all(sample_features_list, "LF7")
        base = generator._generate_base_grid(predictions)

        # Chaque résultat doit être le favori (proba max)
        for i, pred in enumerate(predictions):
            expected = pred["prediction"]
            assert base["resultats"][i] == expected

    def test_base_grid_has_highest_probability(self, generator, sample_features_list):
        predictions = generator._predict_all(sample_features_list, "LF7")
        base = generator._generate_base_grid(predictions)
        variants = generator._generate_variants(predictions, 10)

        # La grille de base devrait avoir la probabilité la plus élevée
        for v in variants:
            assert base["probabilite"] >= v["probabilite"] - 1e-10


# =====================================================
# Tests des variantes
# =====================================================

class TestVariants:
    def test_variants_differ_from_base(self, generator, sample_features_list):
        predictions = generator._predict_all(sample_features_list, "LF7")
        base = generator._generate_base_grid(predictions)
        variants = generator._generate_variants(predictions, 10)

        # Au moins une variante devrait différer de la base
        if variants:
            different = [v for v in variants if v["resultats"] != base["resultats"]]
            assert len(different) > 0

    def test_variants_are_valid(self, generator, sample_features_list):
        predictions = generator._predict_all(sample_features_list, "LF7")
        variants = generator._generate_variants(predictions, 10)

        for v in variants:
            assert len(v["resultats"]) == 7
            assert all(c in "1N2" for c in v["resultats"])
            assert v["probabilite"] > 0
            assert v["confiance"] >= 0

    def test_variants_count_limited(self, generator, sample_features_list):
        predictions = generator._predict_all(sample_features_list, "LF7")
        variants = generator._generate_variants(predictions, 3)
        assert len(variants) <= 3


# =====================================================
# Tests du budget
# =====================================================

class TestBudget:
    def test_budget_1(self, generator, sample_features_list):
        grids = generator.generate(sample_features_list, budget=1)
        assert len(grids) == 1

    def test_budget_large(self, generator, sample_features_list):
        grids = generator.generate(sample_features_list, budget=50)
        # Le nombre de grilles ne peut pas dépasser la combinatoire
        assert len(grids) <= 50


# =====================================================
# Tests de compute_grid_probability
# =====================================================

class TestComputeGridProbability:
    def test_product_of_probas(self):
        predictions = [
            {"probas": {"1": 0.5, "N": 0.3, "2": 0.2}},
            {"probas": {"1": 0.4, "N": 0.3, "2": 0.3}},
            {"probas": {"1": 0.6, "N": 0.2, "2": 0.2}},
        ]

        prob = compute_grid_probability(predictions, "111")
        expected = 0.5 * 0.4 * 0.6
        assert abs(prob - expected) < 1e-10

    def test_mixed_results(self):
        predictions = [
            {"probas": {"1": 0.5, "N": 0.3, "2": 0.2}},
            {"probas": {"1": 0.4, "N": 0.3, "2": 0.3}},
        ]

        prob = compute_grid_probability(predictions, "1N")
        expected = 0.5 * 0.3
        assert abs(prob - expected) < 1e-10

    def test_single_match(self):
        predictions = [
            {"probas": {"1": 0.6, "N": 0.25, "2": 0.15}},
        ]
        prob = compute_grid_probability(predictions, "2")
        assert abs(prob - 0.15) < 1e-10


# =====================================================
# Tests des 3 stratégies
# =====================================================

class TestStrategies:
    def test_prudente_fewer_variants(self, sample_features_list):
        gen_pru = GridGenerator(
            predictor=Predictor(model_path="/nonexistent/path.joblib", strategy="prudente"),
            strategy="prudente",
        )
        gen_aud = GridGenerator(
            predictor=Predictor(model_path="/nonexistent/path.joblib", strategy="audacieuse"),
            strategy="audacieuse",
        )

        grids_pru = gen_pru.generate(sample_features_list, budget=50)
        grids_aud = gen_aud.generate(sample_features_list, budget=50)

        # Audacieuse devrait produire plus de variantes que prudente
        assert len(grids_pru) <= len(grids_aud)

    def test_all_strategies_produce_grids(self, sample_features_list):
        for strategy in ["prudente", "equilibree", "audacieuse"]:
            gen = GridGenerator(
                predictor=Predictor(model_path="/nonexistent/path.joblib", strategy=strategy),
                strategy=strategy,
            )
            grids = gen.generate(sample_features_list, budget=5)
            assert len(grids) >= 1, f"Stratégie {strategy} n'a produit aucune grille"

    def test_all_strategies_include_base_grid(self, sample_features_list, monkeypatch):
        # Désactiver la pondération par profil pour ce test (comportement pré-pondération)
        monkeypatch.setattr(
            "collectors.pronosoft_scraper.fetch_combinaisons_stats",
            lambda gt: {},
        )
        for strategy in ["prudente", "equilibree", "audacieuse"]:
            predictor = Predictor(model_path="/nonexistent/path.joblib", strategy=strategy)
            gen = GridGenerator(predictor=predictor, strategy=strategy)
            predictions = gen._predict_all(sample_features_list, "LF7")
            base = gen._generate_base_grid(predictions)
            grids = gen.generate(sample_features_list, budget=10)

            # Sans pondération profil, la grille de base (favoris) est toujours présente
            base_found = any(g["resultats"] == base["resultats"] for g in grids)
            assert base_found, f"Grille de base absente avec stratégie {strategy}"


# =====================================================
# Tests de compute_grid_expected_value
# =====================================================

class TestExpectedValue:
    def test_positive_ev(self):
        grid = [
            {"prediction": "1", "prob_1": 0.8, "prob_n": 0.1, "prob_2": 0.1},
            {"prediction": "1", "prob_1": 0.7, "prob_n": 0.2, "prob_2": 0.1},
        ]
        ev = compute_grid_expected_value(grid, rapport_moyen=10.0)
        # prob = 0.8 * 0.7 = 0.56, ev = 0.56 * 10 - 1 = 4.6
        assert abs(ev - 4.6) < 0.01

    def test_negative_ev(self):
        grid = [
            {"prediction": "1", "prob_1": 0.33, "prob_n": 0.34, "prob_2": 0.33},
            {"prediction": "1", "prob_1": 0.33, "prob_n": 0.34, "prob_2": 0.33},
            {"prediction": "1", "prob_1": 0.33, "prob_n": 0.34, "prob_2": 0.33},
        ]
        ev = compute_grid_expected_value(grid, rapport_moyen=10.0)
        # prob ≈ 0.33^3 ≈ 0.036, ev ≈ 0.036 * 10 - 1 ≈ -0.64
        assert ev < 0


# =====================================================
# Tests de _compute_grid_profile
# =====================================================

class TestComputeGridProfile:
    def test_balanced_profile(self):
        assert _compute_grid_profile("1N21N21") == "3-2-2"

    def test_all_ones(self):
        assert _compute_grid_profile("1111111") == "7-0-0"

    def test_all_draws(self):
        assert _compute_grid_profile("NNNNNNN") == "0-7-0"

    def test_all_twos(self):
        assert _compute_grid_profile("2222222") == "0-0-7"

    def test_mixed(self):
        assert _compute_grid_profile("1N2") == "1-1-1"


# =====================================================
# Tests optimize_grids avec grid_type (compatibilité)
# =====================================================

class TestOptimizeGridsWithGridType:
    def test_without_grid_type_still_works(self):
        """optimize_grids sans grid_type doit fonctionner comme avant."""
        predictions = [
            {"prediction": "1", "confiance": 0.6,
             "prob_1": 0.6, "prob_n": 0.2, "prob_2": 0.2,
             "probas": {"1": 0.6, "N": 0.2, "2": 0.2}},
            {"prediction": "1", "confiance": 0.5,
             "prob_1": 0.5, "prob_n": 0.3, "prob_2": 0.2,
             "probas": {"1": 0.5, "N": 0.3, "2": 0.2}},
            {"prediction": "2", "confiance": 0.55,
             "prob_1": 0.2, "prob_n": 0.25, "prob_2": 0.55,
             "probas": {"1": 0.2, "N": 0.25, "2": 0.55}},
        ]
        grids = optimize_grids(predictions, budget=5)
        assert len(grids) >= 1
        for g in grids:
            assert "resultats" in g
            assert "probabilite" in g

    def test_with_grid_type_none(self):
        """grid_type=None ne casse pas le comportement."""
        predictions = [
            {"prediction": "1", "confiance": 0.6,
             "prob_1": 0.6, "prob_n": 0.2, "prob_2": 0.2,
             "probas": {"1": 0.6, "N": 0.2, "2": 0.2}},
        ]
        grids = optimize_grids(predictions, budget=3, grid_type=None)
        assert len(grids) >= 1

    def test_with_grid_type_adds_profil_fields(self, monkeypatch):
        """Quand les stats combinaisons sont disponibles, les champs profil sont ajoutés."""
        mock_stats = {"2-0-1": 0.3, "1-1-1": 0.5, "3-0-0": 0.1, "0-0-3": 0.1}
        monkeypatch.setattr(
            "collectors.pronosoft_scraper.fetch_combinaisons_stats",
            lambda gt: mock_stats,
        )

        predictions = [
            {"prediction": "1", "confiance": 0.6,
             "prob_1": 0.6, "prob_n": 0.2, "prob_2": 0.2,
             "probas": {"1": 0.6, "N": 0.2, "2": 0.2}},
            {"prediction": "N", "confiance": 0.4,
             "prob_1": 0.3, "prob_n": 0.4, "prob_2": 0.3,
             "probas": {"1": 0.3, "N": 0.4, "2": 0.3}},
            {"prediction": "2", "confiance": 0.55,
             "prob_1": 0.2, "prob_n": 0.25, "prob_2": 0.55,
             "probas": {"1": 0.2, "N": 0.25, "2": 0.55}},
        ]
        grids = optimize_grids(predictions, budget=10, grid_type="LF7")
        assert len(grids) >= 1
        for g in grids:
            assert "profil" in g
            assert "profil_weight" in g
            assert "score" in g
