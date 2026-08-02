"""Tests pour le prédicteur basé sur les cotes (models/odds_predictor.py)."""

import pytest
import numpy as np

from models.odds_predictor import OddsPredictor, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Tests odds_to_probabilities
# ---------------------------------------------------------------------------

class TestOddsToProbabilities:
    def test_basic_conversion(self):
        p1, pn, p2 = OddsPredictor.odds_to_probabilities(1.50, 4.00, 6.00)
        assert abs(p1 + pn + p2 - 1.0) < 1e-9
        assert p1 > pn > p2  # Favori domicile

    def test_balanced_odds(self):
        p1, pn, p2 = OddsPredictor.odds_to_probabilities(3.00, 3.00, 3.00)
        assert abs(p1 - 1 / 3) < 1e-9
        assert abs(pn - 1 / 3) < 1e-9
        assert abs(p2 - 1 / 3) < 1e-9

    def test_strong_favorite(self):
        p1, pn, p2 = OddsPredictor.odds_to_probabilities(1.10, 8.00, 15.00)
        assert p1 > 0.75
        assert p2 < 0.10

    def test_zero_cote_fallback(self):
        p1, pn, p2 = OddsPredictor.odds_to_probabilities(0, 3.0, 3.0)
        assert abs(p1 - 1 / 3) < 1e-9

    def test_negative_cote_fallback(self):
        p1, pn, p2 = OddsPredictor.odds_to_probabilities(-1.5, 3.0, 3.0)
        assert abs(p1 - 1 / 3) < 1e-9


# ---------------------------------------------------------------------------
# Tests predict_match
# ---------------------------------------------------------------------------

class TestPredictMatch:
    def setup_method(self):
        self.predictor = OddsPredictor()

    def test_output_format(self):
        match = {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00}
        result = self.predictor.predict_match(match)
        assert "prob_1" in result
        assert "prob_n" in result
        assert "prob_2" in result
        assert "prediction" in result
        assert "confiance" in result
        assert "probas" in result

    def test_probabilities_sum_to_one(self):
        match = {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00}
        result = self.predictor.predict_match(match)
        total = result["prob_1"] + result["prob_n"] + result["prob_2"]
        assert abs(total - 1.0) < 1e-6

    def test_prediction_is_argmax(self):
        match = {"cote_1": 1.20, "cote_n": 6.00, "cote_2": 10.00}
        result = self.predictor.predict_match(match)
        assert result["prediction"] == "1"
        assert result["prob_1"] > result["prob_n"]
        assert result["prob_1"] > result["prob_2"]

    def test_away_favorite(self):
        match = {"cote_1": 8.00, "cote_n": 5.00, "cote_2": 1.25}
        result = self.predictor.predict_match(match)
        assert result["prediction"] == "2"

    def test_confiance_positive(self):
        match = {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00}
        result = self.predictor.predict_match(match)
        assert result["confiance"] >= 0

    def test_with_percentages(self):
        match = {
            "cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00,
            "pct_1": 70, "pct_n": 15, "pct_2": 15,
        }
        result = self.predictor.predict_match(match)
        assert result["prediction"] == "1"

    def test_with_cyborg(self):
        match = {
            "cote_1": 2.50, "cote_n": 3.00, "cote_2": 2.80,
            "prono_cyborg": "N",
        }
        result = self.predictor.predict_match(match)
        # Cyborg signal should boost N
        assert result["prob_n"] > 0.20

    def test_with_grid_metrics(self):
        match = {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00}
        metrics = {"difficulty": 3.0, "moyenne_cote_fav": 1.50}
        result = self.predictor.predict_match(match, grid_metrics=metrics)
        assert result["prediction"] == "1"

    def test_missing_cotes_fallback(self):
        match = {}
        result = self.predictor.predict_match(match)
        total = result["prob_1"] + result["prob_n"] + result["prob_2"]
        assert abs(total - 1.0) < 1e-6

    def test_probas_dict_matches_fields(self):
        match = {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00}
        result = self.predictor.predict_match(match)
        assert abs(result["probas"]["1"] - result["prob_1"]) < 1e-9
        assert abs(result["probas"]["N"] - result["prob_n"]) < 1e-9
        assert abs(result["probas"]["2"] - result["prob_2"]) < 1e-9


# ---------------------------------------------------------------------------
# Tests predict_grid
# ---------------------------------------------------------------------------

class TestPredictGrid:
    def test_predict_grid_length(self):
        predictor = OddsPredictor()
        grid_data = {
            "matches": [
                {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00},
                {"cote_1": 2.00, "cote_n": 3.50, "cote_2": 3.50},
                {"cote_1": 3.00, "cote_n": 3.00, "cote_2": 2.50},
            ],
        }
        preds = predictor.predict_grid(grid_data)
        assert len(preds) == 3

    def test_predict_grid_format(self):
        predictor = OddsPredictor()
        grid_data = {
            "matches": [
                {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00,
                 "home": "Team A", "away": "Team B"},
            ],
        }
        preds = predictor.predict_grid(grid_data)
        assert preds[0]["home"] == "Team A"
        assert preds[0]["away"] == "Team B"
        assert "prob_1" in preds[0]

    def test_predict_grid_with_metrics(self):
        predictor = OddsPredictor()
        grid_data = {
            "difficulty": 6.0,
            "moyenne_cote_fav": 1.80,
            "matches": [
                {"cote_1": 1.80, "cote_n": 3.50, "cote_2": 4.00},
            ],
        }
        preds = predictor.predict_grid(grid_data)
        assert len(preds) == 1


# ---------------------------------------------------------------------------
# Tests compute_grid_metrics
# ---------------------------------------------------------------------------

class TestComputeGridMetrics:
    def test_basic(self):
        matches = [
            {"cote_1": 1.50, "cote_n": 3.50, "cote_2": 5.00},
            {"cote_1": 2.00, "cote_n": 3.00, "cote_2": 3.50},
        ]
        metrics = OddsPredictor.compute_grid_metrics(matches)
        assert metrics["somme_cotes_fav"] == round(1.50 + 2.00, 2)
        assert metrics["moyenne_cote_fav"] == round((1.50 + 2.00) / 2, 2)

    def test_empty_matches(self):
        metrics = OddsPredictor.compute_grid_metrics([])
        assert metrics["somme_cotes_fav"] == 0.0


# ---------------------------------------------------------------------------
# Tests strategy selection
# ---------------------------------------------------------------------------

class TestStrategySelection:
    def setup_method(self):
        self.predictor = OddsPredictor()

    def test_secure(self):
        metrics = {"difficulty": 3.0, "moyenne_cote_fav": 1.40}
        assert self.predictor.select_strategy_profile(metrics) == "secure"

    def test_risky_difficulty(self):
        metrics = {"difficulty": 9.0, "moyenne_cote_fav": 1.80}
        assert self.predictor.select_strategy_profile(metrics) == "risky"

    def test_risky_cote(self):
        metrics = {"difficulty": 6.0, "moyenne_cote_fav": 2.50}
        assert self.predictor.select_strategy_profile(metrics) == "risky"

    def test_balanced(self):
        metrics = {"difficulty": 6.0, "moyenne_cote_fav": 1.80}
        assert self.predictor.select_strategy_profile(metrics) == "balanced"

    def test_none_metrics(self):
        assert self.predictor.select_strategy_profile(None) == "balanced"

    def test_missing_keys(self):
        assert self.predictor.select_strategy_profile({}) == "balanced"


# ---------------------------------------------------------------------------
# Tests cyborg signal parsing
# ---------------------------------------------------------------------------

class TestCyborgSignal:
    def setup_method(self):
        self.predictor = OddsPredictor()

    def test_single_1(self):
        signal = self.predictor._parse_cyborg_signal("1")
        assert signal[0] == 0.80
        assert signal[1] == 0.10
        assert signal[2] == 0.10

    def test_single_n(self):
        signal = self.predictor._parse_cyborg_signal("N")
        assert signal[1] == 0.80

    def test_double_n2(self):
        signal = self.predictor._parse_cyborg_signal("N2")
        assert signal[1] == 0.45
        assert signal[2] == 0.45
        assert signal[0] == 0.10

    def test_triple_1n2(self):
        signal = self.predictor._parse_cyborg_signal("1N2")
        assert abs(signal[0] - 1 / 3) < 1e-9

    def test_empty(self):
        signal = self.predictor._parse_cyborg_signal("")
        assert abs(signal[0] - 1 / 3) < 1e-9


# ---------------------------------------------------------------------------
# Tests adjust_probabilities_for_difficulty
# ---------------------------------------------------------------------------

class TestAdjustProbabilities:
    def setup_method(self):
        self.predictor = OddsPredictor()

    def test_secure_boosts_favorite(self):
        p1, pn, p2 = self.predictor.adjust_probabilities_for_difficulty(
            0.6, 0.2, 0.2, None, "secure"
        )
        assert p1 > 0.6  # Boosted

    def test_risky_flattens(self):
        p1, pn, p2 = self.predictor.adjust_probabilities_for_difficulty(
            0.7, 0.15, 0.15, None, "risky"
        )
        assert p1 < 0.7  # Flattened toward uniform

    def test_balanced_no_change(self):
        p1, pn, p2 = self.predictor.adjust_probabilities_for_difficulty(
            0.5, 0.3, 0.2, None, "balanced"
        )
        assert abs(p1 - 0.5) < 1e-9
        assert abs(pn - 0.3) < 1e-9
        assert abs(p2 - 0.2) < 1e-9

    def test_sum_to_one(self):
        for profile in ("secure", "balanced", "risky"):
            p1, pn, p2 = self.predictor.adjust_probabilities_for_difficulty(
                0.5, 0.3, 0.2, None, profile
            )
            assert abs(p1 + pn + p2 - 1.0) < 1e-6
