"""Tests pour le module api (Phase 4) — API REST FastAPI."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.deps import get_predictor, get_db
from models.predictor import Predictor


# =====================================================
# Fixtures
# =====================================================

@pytest.fixture
def mock_predictor():
    """Predictor sans modèle ML pour les tests."""
    return Predictor(model_path="/nonexistent/path.joblib", strategy="equilibree")


@pytest.fixture
def mock_session():
    """Session DB mockée qui ne se connecte pas à la vraie base."""
    session = MagicMock()
    # Pour les requêtes select, retourner des résultats vides
    session.execute.return_value.scalars.return_value.all.return_value = []
    session.execute.return_value.scalar.return_value = None
    return session


@pytest.fixture
def client(mock_predictor, mock_session):
    """TestClient avec dépendances overridées."""

    def override_predictor():
        return mock_predictor

    def override_db():
        yield mock_session

    app.dependency_overrides[get_predictor] = override_predictor
    app.dependency_overrides[get_db] = override_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# =====================================================
# Tests Health Check
# =====================================================

class TestHealthCheck:
    def test_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.json()["status"] == "ok"


# =====================================================
# Tests POST /api/predict
# =====================================================

class TestPredict:
    def test_predict_returns_200(self, client, mock_predictor, mock_session):
        # Mock predict_from_ids pour retourner un résultat valide
        mock_predictor.predict_from_ids = MagicMock(return_value={
            "prob_1": 0.50, "prob_n": 0.25, "prob_2": 0.25,
            "prediction": "1", "confiance": 0.25,
        })

        response = client.post("/api/predict", json={
            "equipe_dom_id": 1,
            "equipe_ext_id": 2,
            "competition_id": 1,
            "date": "2024-01-15",
        })
        assert response.status_code == 200

    def test_predict_response_structure(self, client, mock_predictor):
        mock_predictor.predict_from_ids = MagicMock(return_value={
            "prob_1": 0.50, "prob_n": 0.25, "prob_2": 0.25,
            "prediction": "1", "confiance": 0.25,
        })

        response = client.post("/api/predict", json={
            "equipe_dom_id": 1,
            "equipe_ext_id": 2,
            "competition_id": 1,
            "date": "2024-01-15",
        })

        data = response.json()
        assert "prob_1" in data
        assert "prob_n" in data
        assert "prob_2" in data
        assert "prediction" in data
        assert "confiance" in data
        assert data["prediction"] in ["1", "N", "2"]

    def test_predict_fallback_when_none(self, client, mock_predictor):
        mock_predictor.predict_from_ids = MagicMock(return_value=None)

        response = client.post("/api/predict", json={
            "equipe_dom_id": 1,
            "equipe_ext_id": 2,
            "competition_id": 1,
            "date": "2024-01-15",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "1"
        assert data["confiance"] == 0.0

    def test_predict_invalid_body_returns_422(self, client):
        response = client.post("/api/predict", json={
            "equipe_dom_id": "not_a_number",
        })
        assert response.status_code == 422


# =====================================================
# Tests POST /api/predict/batch
# =====================================================

class TestPredictBatch:
    def test_batch_returns_200(self, client, mock_predictor):
        mock_predictor.predict_from_ids = MagicMock(return_value={
            "prob_1": 0.50, "prob_n": 0.25, "prob_2": 0.25,
            "prediction": "1", "confiance": 0.25,
        })

        response = client.post("/api/predict/batch", json={
            "matches": [
                {"equipe_dom_id": 1, "equipe_ext_id": 2, "competition_id": 1, "date": "2024-01-15"},
                {"equipe_dom_id": 3, "equipe_ext_id": 4, "competition_id": 1, "date": "2024-01-15"},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


# =====================================================
# Tests POST /api/grilles/generate
# =====================================================

class TestGrillesGenerate:
    def test_generate_returns_200(self, client, mock_predictor):
        mock_predictor.predict_from_ids = MagicMock(return_value={
            "prob_1": 0.50, "prob_n": 0.25, "prob_2": 0.25,
            "prediction": "1", "confiance": 0.25,
            "detail_scores": {}, "filtre": True,
        })

        matches = [
            {"equipe_dom_id": i, "equipe_ext_id": i + 10, "competition_id": 1, "date": "2024-01-15"}
            for i in range(1, 8)
        ]

        response = client.post("/api/grilles/generate", json={
            "matches": matches,
            "grid_type": "LF7",
            "budget": 5,
            "strategy": "equilibree",
        })
        assert response.status_code == 200

    def test_generate_response_structure(self, client, mock_predictor):
        mock_predictor.predict_from_ids = MagicMock(return_value={
            "prob_1": 0.50, "prob_n": 0.25, "prob_2": 0.25,
            "prediction": "1", "confiance": 0.25,
            "detail_scores": {}, "filtre": True,
        })

        matches = [
            {"equipe_dom_id": i, "equipe_ext_id": i + 10, "competition_id": 1, "date": "2024-01-15"}
            for i in range(1, 8)
        ]

        response = client.post("/api/grilles/generate", json={
            "matches": matches,
            "grid_type": "LF7",
            "budget": 5,
            "strategy": "equilibree",
        })

        data = response.json()
        assert "grilles" in data
        assert "stats" in data
        assert "nb_grilles" in data["stats"]
        assert "confiance_moyenne" in data["stats"]
        assert len(data["grilles"]) <= 5

        for g in data["grilles"]:
            assert "resultats" in g
            assert "confiance" in g
            assert "probabilite" in g
            assert "matchs" in g

    def test_generate_invalid_strategy(self, client, mock_predictor):
        # Stratégie inconnue ne cause pas d'erreur (fallback equilibree behavior)
        mock_predictor.predict_from_ids = MagicMock(return_value={
            "prob_1": 0.50, "prob_n": 0.25, "prob_2": 0.25,
            "prediction": "1", "confiance": 0.25,
            "detail_scores": {}, "filtre": True,
        })

        matches = [
            {"equipe_dom_id": i, "equipe_ext_id": i + 10, "competition_id": 1, "date": "2024-01-15"}
            for i in range(1, 8)
        ]

        response = client.post("/api/grilles/generate", json={
            "matches": matches,
            "grid_type": "LF7",
            "budget": 5,
            "strategy": "inconnue",
        })
        assert response.status_code == 200


# =====================================================
# Tests GET /api/grilles/history
# =====================================================

class TestGrillesHistory:
    def test_history_returns_200(self, client):
        response = client.get("/api/grilles/history")
        assert response.status_code == 200

    def test_history_returns_list(self, client):
        response = client.get("/api/grilles/history")
        assert isinstance(response.json(), list)

    def test_history_with_type_filter(self, client):
        response = client.get("/api/grilles/history?type_grille=LF7")
        assert response.status_code == 200

    def test_history_with_limit(self, client):
        response = client.get("/api/grilles/history?limit=5")
        assert response.status_code == 200


# =====================================================
# Tests GET /api/stats/distribution
# =====================================================

class TestStatsDistribution:
    def test_distribution_returns_200(self, client):
        with patch("api.app.get_distribution_by_type", return_value={
            "LF7": {"nb_grilles": 50, "moy_1": 3.1, "moy_n": 1.9, "moy_2": 2.0},
        }):
            response = client.get("/api/stats/distribution")
            assert response.status_code == 200


# =====================================================
# Tests de validation des schémas
# =====================================================

class TestSchemaValidation:
    def test_predict_missing_fields(self, client):
        response = client.post("/api/predict", json={})
        assert response.status_code == 422

    def test_predict_wrong_types(self, client):
        response = client.post("/api/predict", json={
            "equipe_dom_id": "abc",
            "equipe_ext_id": 2,
            "competition_id": 1,
            "date": "2024-01-15",
        })
        assert response.status_code == 422

    def test_generate_budget_too_high(self, client):
        matches = [
            {"equipe_dom_id": i, "equipe_ext_id": i + 10, "competition_id": 1, "date": "2024-01-15"}
            for i in range(1, 8)
        ]
        response = client.post("/api/grilles/generate", json={
            "matches": matches,
            "budget": 100,  # Exceeds max of 50
        })
        assert response.status_code == 422

    def test_generate_budget_zero(self, client):
        matches = [
            {"equipe_dom_id": i, "equipe_ext_id": i + 10, "competition_id": 1, "date": "2024-01-15"}
            for i in range(1, 8)
        ]
        response = client.post("/api/grilles/generate", json={
            "matches": matches,
            "budget": 0,  # Below min of 1
        })
        assert response.status_code == 422
