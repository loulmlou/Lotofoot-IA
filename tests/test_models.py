"""Tests pour le module models — OddsPredictor et configuration.

Remplace les anciens tests de scoring/predictor/train qui s'appuyaient
sur les features d'équipe et le modèle ML.
"""

import pytest
import numpy as np

from models.odds_predictor import OddsPredictor, DEFAULT_WEIGHTS
from config.settings import ODDS_MODEL_WEIGHTS, DIFFICULTY_THRESHOLDS


# =====================================================
# Tests de configuration
# =====================================================

class TestConfig:
    def test_weights_sum_to_one(self):
        total = sum(ODDS_MODEL_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Les poids doivent sommer à 1, got {total}"

    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_weight_keys_present(self):
        expected = {"cotes", "consensus", "cyborg", "difficulte", "historique"}
        assert set(ODDS_MODEL_WEIGHTS.keys()) == expected

    def test_difficulty_thresholds_consistent(self):
        t = DIFFICULTY_THRESHOLDS
        assert t["secure_max_difficulty"] < t["risky_min_difficulty"]
        assert t["secure_max_cote_fav"] < t["risky_min_cote_fav"]


# =====================================================
# Tests OddsPredictor complets
# =====================================================

class TestOddsPredictorComplete:
    @pytest.fixture
    def predictor(self):
        return OddsPredictor()

    @pytest.fixture
    def sample_match(self):
        return {
            "cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00,
            "pct_1": 65.0, "pct_n": 20.0, "pct_2": 15.0,
            "prono_cyborg": "1",
        }

    def test_predict_match_returns_valid_output(self, predictor, sample_match):
        result = predictor.predict_match(sample_match)

        assert "prob_1" in result
        assert "prob_n" in result
        assert "prob_2" in result
        assert "prediction" in result
        assert "confiance" in result
        assert "probas" in result

        total = result["prob_1"] + result["prob_n"] + result["prob_2"]
        assert abs(total - 1.0) < 0.001
        assert result["prediction"] in ["1", "N", "2"]
        assert result["confiance"] >= 0

    def test_strong_favorite_predicts_correctly(self, predictor):
        match = {"cote_1": 1.10, "cote_n": 8.0, "cote_2": 15.0,
                 "pct_1": 90, "pct_n": 5, "pct_2": 5,
                 "prono_cyborg": "1"}
        result = predictor.predict_match(match)
        assert result["prediction"] == "1"
        assert result["prob_1"] > 0.65

    def test_away_favorite(self, predictor):
        match = {"cote_1": 10.0, "cote_n": 6.0, "cote_2": 1.15,
                 "pct_1": 5, "pct_n": 10, "pct_2": 85,
                 "prono_cyborg": "2"}
        result = predictor.predict_match(match)
        assert result["prediction"] == "2"

    def test_no_data_returns_near_uniform(self, predictor):
        result = predictor.predict_match({})
        total = result["prob_1"] + result["prob_n"] + result["prob_2"]
        assert abs(total - 1.0) < 0.001

    def test_all_strategies_produce_valid_output(self, sample_match):
        for strategy in ["prudente", "equilibree", "audacieuse"]:
            pred = OddsPredictor(strategy=strategy)
            result = pred.predict_match(sample_match)
            assert result["prediction"] in ["1", "N", "2"]
            total = result["prob_1"] + result["prob_n"] + result["prob_2"]
            assert abs(total - 1.0) < 0.001

    def test_custom_weights(self, sample_match):
        weights = {
            "cotes": 0.90, "consensus": 0.025,
            "cyborg": 0.025, "difficulte": 0.025, "historique": 0.025,
        }
        pred = OddsPredictor(weights=weights)
        result = pred.predict_match(sample_match)
        assert result["prediction"] in ["1", "N", "2"]

    def test_predict_grid_returns_correct_count(self, predictor):
        grid_data = {
            "matches": [
                {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00},
                {"cote_1": 2.00, "cote_n": 3.50, "cote_2": 3.50},
                {"cote_1": 3.00, "cote_n": 3.00, "cote_2": 2.50},
                {"cote_1": 1.80, "cote_n": 3.40, "cote_2": 4.20},
                {"cote_1": 2.20, "cote_n": 3.20, "cote_2": 3.00},
                {"cote_1": 1.35, "cote_n": 4.50, "cote_2": 7.00},
                {"cote_1": 4.00, "cote_n": 3.50, "cote_2": 1.80},
            ],
        }
        preds = predictor.predict_grid(grid_data)
        assert len(preds) == 7
        for p in preds:
            assert "prob_1" in p
            assert "prediction" in p

    def test_probas_dict_consistency(self, predictor, sample_match):
        result = predictor.predict_match(sample_match)
        assert abs(result["probas"]["1"] - result["prob_1"]) < 1e-9
        assert abs(result["probas"]["N"] - result["prob_n"]) < 1e-9
        assert abs(result["probas"]["2"] - result["prob_2"]) < 1e-9
