"""Tests pour le module generator — génération de grilles optimisées basée cotes."""

import pytest

from models.odds_predictor import OddsPredictor
from generator.grid_generator import GridGenerator
from generator.optimizer import (
    optimize_grids, compute_grid_probability,
    compute_grid_expected_value, _compute_grid_profile,
)


# =====================================================
# Fixture : données de matchs basées sur les cotes
# =====================================================

@pytest.fixture
def sample_grid_data():
    """Retourne un dict grid_data simulant une grille LF7 avec cotes."""
    return {
        "difficulty": 6.5,
        "matches": [
            {"cote_1": 1.35, "cote_n": 4.50, "cote_2": 7.00,
             "pct_1": 70, "pct_n": 18, "pct_2": 12,
             "prono_cyborg": "1", "home": "Team A", "away": "Team B"},
            {"cote_1": 2.50, "cote_n": 3.20, "cote_2": 2.60,
             "pct_1": 35, "pct_n": 30, "pct_2": 35,
             "prono_cyborg": "N2", "home": "Team C", "away": "Team D"},
            {"cote_1": 5.00, "cote_n": 4.00, "cote_2": 1.40,
             "pct_1": 12, "pct_n": 20, "pct_2": 68,
             "prono_cyborg": "2", "home": "Team E", "away": "Team F"},
            {"cote_1": 1.80, "cote_n": 3.50, "cote_2": 4.20,
             "pct_1": 50, "pct_n": 25, "pct_2": 25,
             "prono_cyborg": "1", "home": "Team G", "away": "Team H"},
            {"cote_1": 2.20, "cote_n": 3.20, "cote_2": 3.00,
             "pct_1": 40, "pct_n": 28, "pct_2": 32,
             "prono_cyborg": "12", "home": "Team I", "away": "Team J"},
            {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 5.50,
             "pct_1": 60, "pct_n": 22, "pct_2": 18,
             "prono_cyborg": "1", "home": "Team K", "away": "Team L"},
            {"cote_1": 3.50, "cote_n": 3.30, "cote_2": 2.00,
             "pct_1": 25, "pct_n": 30, "pct_2": 45,
             "prono_cyborg": "2", "home": "Team M", "away": "Team N"},
        ],
    }


@pytest.fixture
def predictor():
    """OddsPredictor par défaut."""
    return OddsPredictor(strategy="equilibree")


@pytest.fixture
def generator(predictor):
    """GridGenerator avec predictor de test."""
    return GridGenerator(predictor=predictor, strategy="equilibree")


# =====================================================
# Tests de GridGenerator.generate()
# =====================================================

class TestGridGeneratorGenerate:
    def test_returns_list(self, generator, sample_grid_data):
        grids = generator.generate(sample_grid_data, grid_type="LF7", budget=5)
        assert isinstance(grids, list)

    def test_returns_correct_count(self, generator, sample_grid_data):
        grids = generator.generate(sample_grid_data, grid_type="LF7", budget=5)
        assert len(grids) <= 5
        assert len(grids) >= 1

    def test_grid_structure(self, generator, sample_grid_data):
        grids = generator.generate(sample_grid_data, grid_type="LF7", budget=3)
        for g in grids:
            assert "resultats" in g
            assert "confiance" in g
            assert "probabilite" in g
            assert "matchs" in g
            assert len(g["resultats"]) == 7
            assert all(c in "1N2" for c in g["resultats"])

    def test_budget_respected(self, generator, sample_grid_data):
        for budget in [1, 3, 5, 10]:
            grids = generator.generate(sample_grid_data, grid_type="LF7", budget=budget)
            assert len(grids) <= budget

    def test_empty_matches(self, generator):
        grids = generator.generate({"matches": []}, grid_type="LF7", budget=5)
        assert grids == []


# =====================================================
# Tests des stratégies
# =====================================================

class TestStrategies:
    def test_prudente_fewer_variants(self, sample_grid_data):
        gen_pru = GridGenerator(
            predictor=OddsPredictor(strategy="prudente"),
            strategy="prudente",
        )
        gen_aud = GridGenerator(
            predictor=OddsPredictor(strategy="audacieuse"),
            strategy="audacieuse",
        )

        grids_pru = gen_pru.generate(sample_grid_data, budget=50)
        grids_aud = gen_aud.generate(sample_grid_data, budget=50)

        # Audacieuse devrait produire au moins autant de variantes que prudente
        assert len(grids_pru) <= len(grids_aud)

    def test_all_strategies_produce_grids(self, sample_grid_data):
        for strategy in ["prudente", "equilibree", "audacieuse"]:
            gen = GridGenerator(
                predictor=OddsPredictor(strategy=strategy),
                strategy=strategy,
            )
            grids = gen.generate(sample_grid_data, budget=5)
            assert len(grids) >= 1, f"Stratégie {strategy} n'a produit aucune grille"


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
# Tests de compute_grid_expected_value
# =====================================================

class TestExpectedValue:
    def test_positive_ev(self):
        grid = [
            {"prediction": "1", "prob_1": 0.8, "prob_n": 0.1, "prob_2": 0.1},
            {"prediction": "1", "prob_1": 0.7, "prob_n": 0.2, "prob_2": 0.1},
        ]
        ev = compute_grid_expected_value(grid, rapport_moyen=10.0)
        expected = 0.8 * 0.7 * 10.0 - 1.0
        assert abs(ev - expected) < 0.01

    def test_negative_ev(self):
        grid = [
            {"prediction": "1", "prob_1": 0.33, "prob_n": 0.34, "prob_2": 0.33},
            {"prediction": "1", "prob_1": 0.33, "prob_n": 0.34, "prob_2": 0.33},
            {"prediction": "1", "prob_1": 0.33, "prob_n": 0.34, "prob_2": 0.33},
        ]
        ev = compute_grid_expected_value(grid, rapport_moyen=10.0)
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
# Tests optimize_grids avec grid_metrics
# =====================================================

class TestOptimizeGridsWithMetrics:
    def test_without_grid_type_still_works(self):
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

    def test_with_grid_metrics_difficulty(self):
        predictions = [
            {"prediction": "1", "confiance": 0.6,
             "prob_1": 0.6, "prob_n": 0.2, "prob_2": 0.2,
             "probas": {"1": 0.6, "N": 0.2, "2": 0.2}},
            {"prediction": "N", "confiance": 0.4,
             "prob_1": 0.3, "prob_n": 0.4, "prob_2": 0.3,
             "probas": {"1": 0.3, "N": 0.4, "2": 0.3}},
        ]
        grids = optimize_grids(
            predictions, budget=5,
            grid_metrics={"difficulty": 9.0},
        )
        assert len(grids) >= 1

    def test_with_grid_type_adds_profil_fields(self, monkeypatch):
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
