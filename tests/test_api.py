"""Tests pour le module api — API REST FastAPI (basée cotes)."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.deps import get_predictor, get_db
from models.odds_predictor import OddsPredictor


# =====================================================
# Fixtures
# =====================================================

@pytest.fixture
def mock_predictor():
    """OddsPredictor pour les tests."""
    return OddsPredictor(strategy="equilibree")


@pytest.fixture
def mock_session():
    """Session DB mockée qui ne se connecte pas à la vraie base."""
    session = MagicMock()
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
    def test_predict_returns_200(self, client):
        response = client.post("/api/predict", json={
            "cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00,
        })
        assert response.status_code == 200

    def test_predict_response_structure(self, client):
        response = client.post("/api/predict", json={
            "cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00,
            "pct_1": 65, "pct_n": 20, "pct_2": 15,
            "prono_cyborg": "1",
        })

        data = response.json()
        assert "prob_1" in data
        assert "prob_n" in data
        assert "prob_2" in data
        assert "prediction" in data
        assert "confiance" in data
        assert data["prediction"] in ["1", "N", "2"]

    def test_predict_strong_favorite(self, client):
        response = client.post("/api/predict", json={
            "cote_1": 1.10, "cote_n": 8.00, "cote_2": 15.00,
        })
        data = response.json()
        assert data["prediction"] == "1"
        assert data["prob_1"] > 0.5

    def test_predict_empty_body(self, client):
        response = client.post("/api/predict", json={})
        assert response.status_code == 200


# =====================================================
# Tests POST /api/predict/batch
# =====================================================

class TestPredictBatch:
    def test_batch_returns_200(self, client):
        response = client.post("/api/predict/batch", json={
            "matches": [
                {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00},
                {"cote_1": 2.00, "cote_n": 3.50, "cote_2": 3.50},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


# =====================================================
# Tests POST /api/grilles/generate
# =====================================================

class TestGrillesGenerate:
    def test_generate_returns_200(self, client):
        matches = [
            {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00}
            for _ in range(7)
        ]

        response = client.post("/api/grilles/generate", json={
            "matches": matches,
            "grid_type": "LF7",
            "budget": 5,
            "strategy": "equilibree",
        })
        assert response.status_code == 200

    def test_generate_response_structure(self, client):
        matches = [
            {"cote_1": 1.50 + i * 0.2, "cote_n": 3.50, "cote_2": 4.00 + i * 0.3}
            for i in range(7)
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

    def test_generate_invalid_strategy(self, client):
        matches = [
            {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00}
            for _ in range(7)
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
        response = client.get("/api/stats/distribution")
        assert response.status_code == 200


# =====================================================
# Tests de validation des schémas
# =====================================================

class TestSchemaValidation:
    def test_generate_budget_too_high(self, client):
        matches = [
            {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00}
            for _ in range(7)
        ]
        response = client.post("/api/grilles/generate", json={
            "matches": matches,
            "budget": 100,  # Exceeds max of 50
        })
        assert response.status_code == 422

    def test_generate_budget_zero(self, client):
        matches = [
            {"cote_1": 1.50, "cote_n": 4.00, "cote_2": 6.00}
            for _ in range(7)
        ]
        response = client.post("/api/grilles/generate", json={
            "matches": matches,
            "budget": 0,  # Below min of 1
        })
        assert response.status_code == 422
