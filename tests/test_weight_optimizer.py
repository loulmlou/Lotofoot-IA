"""Tests pour l'optimiseur de poids (models/weight_optimizer.py)."""

import json
import os
import tempfile

import pytest

from models.weight_optimizer import WeightOptimizer
from models.odds_predictor import DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_grids(n_grids=5):
    """Crée des grilles synthétiques pour les tests."""
    grids = []
    for i in range(n_grids):
        matches = []
        resultats = ""
        for j in range(7):
            # Strong favorite that wins
            if j % 3 == 0:
                matches.append({
                    "cote_1": 1.30, "cote_n": 4.50, "cote_2": 7.00,
                    "pct_1": 70, "pct_n": 18, "pct_2": 12,
                    "prono_cyborg": "1",
                })
                resultats += "1"
            elif j % 3 == 1:
                matches.append({
                    "cote_1": 2.50, "cote_n": 3.20, "cote_2": 2.60,
                    "pct_1": 35, "pct_n": 30, "pct_2": 35,
                    "prono_cyborg": "N",
                })
                resultats += "N"
            else:
                matches.append({
                    "cote_1": 5.00, "cote_n": 4.00, "cote_2": 1.40,
                    "pct_1": 12, "pct_n": 20, "pct_2": 68,
                    "prono_cyborg": "2",
                })
                resultats += "2"

        grids.append({
            "grid_number": i + 1,
            "matches": matches,
            "resultats": resultats,
            "difficulty": 5.0 + i * 0.5,
            "rapports": {
                "7_sur_7": {"gagnants": 10, "montant": 500.0},
            },
        })
    return grids


@pytest.fixture
def synthetic_history_path():
    """Crée un fichier JSON d'historique synthétique temporaire."""
    grids = _make_synthetic_grids(10)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
    ) as f:
        json.dump(grids, f, ensure_ascii=False)
        path = f.name

    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# Tests evaluate_weights
# ---------------------------------------------------------------------------

class TestEvaluateWeights:
    def test_default_weights_returns_value(self, synthetic_history_path):
        opt = WeightOptimizer(synthetic_history_path)
        score = opt.evaluate_weights(DEFAULT_WEIGHTS, metric="accuracy")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_correct_6_plus_metric(self, synthetic_history_path):
        opt = WeightOptimizer(synthetic_history_path)
        score = opt.evaluate_weights(DEFAULT_WEIGHTS, metric="correct_6_plus")
        assert isinstance(score, float)
        assert score >= 0

    def test_correct_7_metric(self, synthetic_history_path):
        opt = WeightOptimizer(synthetic_history_path)
        score = opt.evaluate_weights(DEFAULT_WEIGHTS, metric="correct_7")
        assert isinstance(score, float)
        assert score >= 0

    def test_empty_grids(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
        ) as f:
            json.dump([], f)
            path = f.name

        try:
            opt = WeightOptimizer(path)
            score = opt.evaluate_weights(DEFAULT_WEIGHTS, metric="accuracy")
            assert score == 0.0
        finally:
            os.unlink(path)

    def test_custom_grids_subset(self, synthetic_history_path):
        opt = WeightOptimizer(synthetic_history_path)
        subset = opt.grids[:3]
        score = opt.evaluate_weights(DEFAULT_WEIGHTS, grids=subset, metric="accuracy")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Tests optimize_global
# ---------------------------------------------------------------------------

class TestOptimizeGlobal:
    def test_returns_valid_weights(self, synthetic_history_path):
        opt = WeightOptimizer(synthetic_history_path)
        result = opt.optimize_global(metric="accuracy", n_restarts=2)
        assert "weights" in result
        assert "score" in result
        weights = result["weights"]
        assert all(v >= 0 for v in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_returns_score(self, synthetic_history_path):
        opt = WeightOptimizer(synthetic_history_path)
        result = opt.optimize_global(metric="accuracy", n_restarts=1)
        assert result["score"] >= 0


# ---------------------------------------------------------------------------
# Tests get_current_weights
# ---------------------------------------------------------------------------

class TestGetCurrentWeights:
    def test_returns_weights(self, synthetic_history_path):
        opt = WeightOptimizer(synthetic_history_path)
        weights = opt.get_current_weights(window_size=5)
        assert isinstance(weights, dict)
        assert all(k in weights for k in DEFAULT_WEIGHTS)
        assert all(v >= 0 for v in weights.values())

    def test_empty_history(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
        ) as f:
            json.dump([], f)
            path = f.name

        try:
            opt = WeightOptimizer(path)
            weights = opt.get_current_weights()
            assert weights == DEFAULT_WEIGHTS
        finally:
            os.unlink(path)
