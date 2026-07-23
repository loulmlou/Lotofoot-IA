"""Tests pour le module models (Phase 3) — entraînement ML, scoring, prédicteur."""

import os
import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Match, Equipe, Competition, Cote


# =====================================================
# Fixture : dataset synthétique pour les tests
# =====================================================

@pytest.fixture
def sample_dataset():
    """Crée un DataFrame synthétique imitant le format de build_dataset()."""
    np.random.seed(42)
    n = 200

    dates = pd.date_range("2022-01-01", periods=n, freq="3D")
    resultats = np.random.choice(["1", "N", "2"], size=n, p=[0.44, 0.27, 0.29])

    df = pd.DataFrame({
        "match_id": range(1, n + 1),
        "date": dates,
        "equipe_dom_id": np.random.randint(1, 10, n),
        "equipe_ext_id": np.random.randint(1, 10, n),
        "competition_id": np.ones(n, dtype=int),
        "resultat": resultats,
        # Forme
        "dom_forme_pts": np.random.uniform(3, 15, n),
        "dom_forme_ratio_vic": np.random.uniform(0, 1, n),
        "dom_forme_buts_m": np.random.uniform(0.5, 3.0, n),
        "dom_forme_buts_e": np.random.uniform(0.3, 2.5, n),
        "dom_forme_nb": np.full(n, 5),
        "ext_forme_pts": np.random.uniform(3, 15, n),
        "ext_forme_ratio_vic": np.random.uniform(0, 1, n),
        "ext_forme_buts_m": np.random.uniform(0.5, 3.0, n),
        "ext_forme_buts_e": np.random.uniform(0.3, 2.5, n),
        "ext_forme_nb": np.full(n, 5),
        # H2H
        "h2h_nb": np.random.randint(0, 6, n),
        "h2h_pct_dom": np.random.uniform(0, 0.6, n),
        "h2h_pct_nul": np.random.uniform(0, 0.4, n),
        "h2h_pct_ext": np.random.uniform(0, 0.5, n),
        # Classement
        "dom_position": np.random.randint(1, 20, n).astype(float),
        "ext_position": np.random.randint(1, 20, n).astype(float),
        "diff_position": np.random.randint(-18, 18, n).astype(float),
        "dom_classement_pts": np.random.uniform(5, 60, n),
        "ext_classement_pts": np.random.uniform(5, 60, n),
        # Buts
        "dom_moy_buts_m": np.random.uniform(0.5, 3.0, n),
        "dom_moy_buts_e": np.random.uniform(0.3, 2.0, n),
        "dom_over25": np.random.uniform(0.2, 0.8, n),
        "ext_moy_buts_m": np.random.uniform(0.5, 3.0, n),
        "ext_moy_buts_e": np.random.uniform(0.3, 2.0, n),
        "ext_over25": np.random.uniform(0.2, 0.8, n),
        # Cotes
        "prob_1": np.random.uniform(0.25, 0.65, n),
        "prob_n": np.random.uniform(0.15, 0.35, n),
        "prob_2": np.random.uniform(0.10, 0.50, n),
        "cote_surprise": np.random.uniform(0.3, 0.7, n),
        "cote_1": np.random.uniform(1.2, 5.0, n),
        "cote_n": np.random.uniform(2.5, 5.0, n),
        "cote_2": np.random.uniform(1.5, 8.0, n),
        # Contexte
        "dom_jours_repos": np.random.randint(3, 14, n).astype(float),
        "ext_jours_repos": np.random.randint(3, 14, n).astype(float),
        "avantage_dom_ligue": np.random.uniform(0.35, 0.55, n),
    })

    # Introduire quelques NaN pour tester l'imputation
    for col in ["dom_position", "ext_position", "diff_position"]:
        mask = np.random.random(n) < 0.3
        df.loc[mask, col] = np.nan

    return df


@pytest.fixture
def sample_features():
    """Retourne un dict de features pour un match typique."""
    return {
        "dom_forme_pts": 10,
        "dom_forme_ratio_vic": 0.6,
        "dom_forme_buts_m": 1.8,
        "dom_forme_buts_e": 0.8,
        "dom_forme_nb": 5,
        "ext_forme_pts": 7,
        "ext_forme_ratio_vic": 0.4,
        "ext_forme_buts_m": 1.2,
        "ext_forme_buts_e": 1.4,
        "ext_forme_nb": 5,
        "h2h_nb": 3,
        "h2h_pct_dom": 0.5,
        "h2h_pct_nul": 0.2,
        "h2h_pct_ext": 0.3,
        "dom_position": 3.0,
        "ext_position": 10.0,
        "diff_position": -7.0,
        "dom_classement_pts": 45.0,
        "ext_classement_pts": 28.0,
        "dom_moy_buts_m": 1.9,
        "dom_moy_buts_e": 0.7,
        "dom_over25": 0.6,
        "ext_moy_buts_m": 1.3,
        "ext_moy_buts_e": 1.5,
        "ext_over25": 0.5,
        "prob_1": 0.55,
        "prob_n": 0.25,
        "prob_2": 0.20,
        "cote_surprise": 0.55,
        "cote_1": 1.8,
        "cote_n": 3.5,
        "cote_2": 4.5,
        "dom_jours_repos": 7.0,
        "ext_jours_repos": 4.0,
        "avantage_dom_ligue": 0.46,
    }


# =====================================================
# Tests du prétraitement (models/train.py)
# =====================================================

class TestPreprocessFeatures:
    def test_no_nan_after_imputation(self, sample_dataset):
        from models.train import preprocess_features

        X, y, feature_names, preprocessor = preprocess_features(sample_dataset)

        assert not np.isnan(X).any(), "NaN détectés après imputation"

    def test_shapes_correct(self, sample_dataset):
        from models.train import preprocess_features

        X, y, feature_names, preprocessor = preprocess_features(sample_dataset)

        assert X.shape[0] == len(sample_dataset)
        assert X.shape[1] == len(feature_names)
        assert len(y) == len(sample_dataset)

    def test_labels_encoded(self, sample_dataset):
        from models.train import preprocess_features

        _, y, _, _ = preprocess_features(sample_dataset)

        assert set(y).issubset({0, 1, 2})

    def test_standardized(self, sample_dataset):
        from models.train import preprocess_features

        X, _, _, _ = preprocess_features(sample_dataset)

        # Les colonnes devraient être standardisées (mean ~0, std ~1)
        col_means = X.mean(axis=0)
        col_stds = X.std(axis=0)

        assert np.allclose(col_means, 0, atol=0.1)
        # Les colonnes constantes auront std=0 après standardisation
        non_constant = col_stds > 0.01
        assert np.allclose(col_stds[non_constant], 1, atol=0.3)

    def test_apply_preprocessor(self, sample_dataset):
        from models.train import preprocess_features, _apply_preprocessor

        X, y, feature_cols, preprocessor = preprocess_features(sample_dataset)
        X2, y2 = _apply_preprocessor(sample_dataset, preprocessor, feature_cols)

        assert X2.shape == X.shape
        assert np.allclose(X, X2)


# =====================================================
# Tests de l'entraînement XGBoost/LightGBM
# =====================================================

class TestTraining:
    def test_xgboost_trains(self, sample_dataset):
        from models.train import preprocess_features, train_xgboost, evaluate_model
        from analysis.dataset_builder import split_by_date

        train_df, val_df, test_df = split_by_date(
            sample_dataset, "2022-08-01", "2022-12-01"
        )

        X_train, y_train, features, preprocessor = preprocess_features(train_df)

        available = [c for c in features if c in val_df.columns]
        X_val = preprocessor.transform(val_df[available])
        y_val = val_df["resultat"].map({"1": 0, "N": 1, "2": 2}).values
        X_test = preprocessor.transform(test_df[available])
        y_test = test_df["resultat"].map({"1": 0, "N": 1, "2": 2}).values

        model = train_xgboost(X_train, y_train, X_val, y_val, search_params=False)

        # Le modèle doit pouvoir prédire
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1, 2})

        # Accuracy > random baseline (33%)
        result = evaluate_model(model, X_test, y_test)
        assert result["accuracy"] > 0.20  # Au moins mieux que le pur hasard

    def test_lightgbm_trains(self, sample_dataset):
        from models.train import preprocess_features, train_lightgbm, evaluate_model
        from analysis.dataset_builder import split_by_date

        train_df, val_df, test_df = split_by_date(
            sample_dataset, "2022-08-01", "2022-12-01"
        )

        X_train, y_train, features, preprocessor = preprocess_features(train_df)

        available = [c for c in features if c in val_df.columns]
        X_val = preprocessor.transform(val_df[available])
        y_val = val_df["resultat"].map({"1": 0, "N": 1, "2": 2}).values
        X_test = preprocessor.transform(test_df[available])
        y_test = test_df["resultat"].map({"1": 0, "N": 1, "2": 2}).values

        model = train_lightgbm(X_train, y_train, X_val, y_val, search_params=False)

        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

        result = evaluate_model(model, X_test, y_test)
        assert result["accuracy"] > 0.20

    def test_save_load_model(self, sample_dataset, tmp_path):
        from models.train import preprocess_features, train_xgboost, save_model, load_model
        from analysis.dataset_builder import split_by_date

        train_df, val_df, _ = split_by_date(
            sample_dataset, "2022-08-01", "2022-12-01"
        )

        X_train, y_train, feature_cols, preprocessor = preprocess_features(train_df)

        available = [c for c in feature_cols if c in val_df.columns]
        X_val = preprocessor.transform(val_df[available])
        y_val = val_df["resultat"].map({"1": 0, "N": 1, "2": 2}).values

        model = train_xgboost(X_train, y_train, X_val, y_val, search_params=False)

        path = str(tmp_path)
        save_model(model, preprocessor, feature_cols, path=path, name="test_model")

        filepath = os.path.join(path, "test_model.joblib")
        assert os.path.exists(filepath)

        artifact = load_model(filepath)
        assert "model" in artifact
        assert "preprocessor" in artifact
        assert "feature_cols" in artifact
        assert "trained_at" in artifact

        # Le modèle chargé doit produire les mêmes prédictions
        preds_orig = model.predict(X_val)
        preds_loaded = artifact["model"].predict(X_val)
        assert np.array_equal(preds_orig, preds_loaded)


# =====================================================
# Tests du scoring pondéré (models/scoring.py)
# =====================================================

class TestScoring:
    def test_weights_sum_to_one(self):
        from config.settings import SCORING_WEIGHTS

        total = sum(SCORING_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Les poids doivent sommer à 1, got {total}"

    def test_score_cotes_normalized(self, sample_features):
        from models.scoring import score_cotes

        result = score_cotes(sample_features)

        assert len(result) == 3
        assert abs(result.sum() - 1.0) < 0.001

    def test_score_cotes_no_data(self):
        from models.scoring import score_cotes

        result = score_cotes({})
        assert abs(result.sum() - 1.0) < 0.001
        # Sans données: distribution uniforme
        assert np.allclose(result, [1/3, 1/3, 1/3], atol=0.01)

    def test_score_forme_normalized(self, sample_features):
        from models.scoring import score_forme

        result = score_forme(sample_features)
        assert len(result) == 3
        assert abs(result.sum() - 1.0) < 0.001

    def test_score_classement_normalized(self, sample_features):
        from models.scoring import score_classement

        result = score_classement(sample_features)
        assert len(result) == 3
        assert abs(result.sum() - 1.0) < 0.001

    def test_score_historique_normalized(self, sample_features):
        from models.scoring import score_historique

        result = score_historique(sample_features)
        assert len(result) == 3
        assert abs(result.sum() - 1.0) < 0.001

    def test_score_historique_no_h2h(self):
        from models.scoring import score_historique

        result = score_historique({"h2h_nb": 0})
        assert abs(result.sum() - 1.0) < 0.001

    def test_score_lotofoot_normalized(self, sample_features):
        from models.scoring import score_lotofoot

        for grid_type in ["LF7", "LF8", "LF15", "LF12", None]:
            result = score_lotofoot(sample_features, grid_type=grid_type)
            assert len(result) == 3
            assert abs(result.sum() - 1.0) < 0.001

    def test_score_modele_ia_no_model(self, sample_features):
        from models.scoring import score_modele_ia

        result = score_modele_ia(sample_features, model=None)
        assert abs(result.sum() - 1.0) < 0.001
        # Sans modèle: uniforme
        assert np.allclose(result, [1/3, 1/3, 1/3], atol=0.01)

    def test_score_contexte_normalized(self, sample_features):
        from models.scoring import score_contexte

        result = score_contexte(sample_features)
        assert len(result) == 3
        assert abs(result.sum() - 1.0) < 0.001

    def test_compute_final_score_structure(self, sample_features):
        from models.scoring import compute_final_score

        result = compute_final_score(sample_features)

        assert "prob_1" in result
        assert "prob_n" in result
        assert "prob_2" in result
        assert "prediction" in result
        assert "confiance" in result
        assert "detail_scores" in result

        # Probabilités normalisées
        total = result["prob_1"] + result["prob_n"] + result["prob_2"]
        assert abs(total - 1.0) < 0.001

        # Prédiction valide
        assert result["prediction"] in ["1", "N", "2"]

        # Confiance positive
        assert result["confiance"] >= 0

    def test_compute_final_score_custom_weights(self, sample_features):
        from models.scoring import compute_final_score

        # Poids personnalisés (que les cotes)
        weights = {
            "cotes": 1.0,
            "forme": 0.0,
            "classement": 0.0,
            "historique": 0.0,
            "stats_lotofoot": 0.0,
            "modele_ia": 0.0,
            "contexte": 0.0,
        }

        result = compute_final_score(sample_features, weights=weights)
        total = result["prob_1"] + result["prob_n"] + result["prob_2"]
        assert abs(total - 1.0) < 0.001

    def test_all_scores_positive(self, sample_features):
        from models.scoring import compute_final_score

        result = compute_final_score(sample_features)
        assert result["prob_1"] >= 0
        assert result["prob_n"] >= 0
        assert result["prob_2"] >= 0

    def test_detail_scores_present(self, sample_features):
        from models.scoring import compute_final_score

        result = compute_final_score(sample_features)
        detail = result["detail_scores"]

        expected_components = ["cotes", "forme", "classement", "historique",
                               "stats_lotofoot", "modele_ia", "contexte"]
        for comp in expected_components:
            assert comp in detail, f"Composant {comp} manquant dans detail_scores"
            assert "prob_1" in detail[comp]
            assert "prob_n" in detail[comp]
            assert "prob_2" in detail[comp]


# =====================================================
# Tests du prédicteur (models/predictor.py)
# =====================================================

class TestPredictor:
    def test_predictor_init_no_model(self):
        from models.predictor import Predictor

        predictor = Predictor(model_path="/nonexistent/path.joblib")
        assert predictor.model is None
        assert predictor.strategy == "equilibree"

    def test_predictor_predict_match(self, sample_features):
        from models.predictor import Predictor

        predictor = Predictor(model_path="/nonexistent/path.joblib")
        result = predictor.predict_match(sample_features)

        assert result is not None
        assert result["prediction"] in ["1", "N", "2"]
        assert 0 <= result["prob_1"] <= 1
        assert 0 <= result["prob_n"] <= 1
        assert 0 <= result["prob_2"] <= 1
        assert "filtre" in result

    def test_predictor_batch(self, sample_features):
        from models.predictor import Predictor

        predictor = Predictor(model_path="/nonexistent/path.joblib")
        matches = [sample_features, sample_features, sample_features]

        df = predictor.predict_batch(matches)

        assert len(df) == 3
        assert "prediction" in df.columns
        assert "confiance" in df.columns
        assert "filtre" in df.columns

    def test_strategy_prudente_filters(self, sample_features):
        from models.predictor import Predictor, STRATEGY_THRESHOLDS

        predictor = Predictor(
            model_path="/nonexistent/path.joblib",
            strategy="prudente",
        )
        assert predictor.confidence_threshold == STRATEGY_THRESHOLDS["prudente"]

        result = predictor.predict_match(sample_features)
        # Le filtre est basé sur la confiance vs le seuil
        assert result["filtre"] == (result["confiance"] >= predictor.confidence_threshold)

    def test_strategy_audacieuse(self, sample_features):
        from models.predictor import Predictor

        predictor = Predictor(
            model_path="/nonexistent/path.joblib",
            strategy="audacieuse",
        )
        assert predictor.confidence_threshold == 0.0

        result = predictor.predict_match(sample_features)
        # Audacieuse accepte tout
        assert result["filtre"] is True

    def test_predictor_with_trained_model(self, sample_dataset, tmp_path):
        from models.train import preprocess_features, train_xgboost, save_model
        from models.predictor import Predictor
        from analysis.dataset_builder import split_by_date

        train_df, val_df, _ = split_by_date(
            sample_dataset, "2022-08-01", "2022-12-01"
        )

        X_train, y_train, feature_cols, preprocessor = preprocess_features(train_df)
        available = [c for c in feature_cols if c in val_df.columns]
        X_val = preprocessor.transform(val_df[available])
        y_val = val_df["resultat"].map({"1": 0, "N": 1, "2": 2}).values

        model = train_xgboost(X_train, y_train, X_val, y_val, search_params=False)

        path = str(tmp_path)
        save_model(model, preprocessor, feature_cols, path=path, name="xgboost")

        model_path = os.path.join(path, "xgboost.joblib")
        predictor = Predictor(model_path=model_path)

        assert predictor.model is not None

        # Prédire avec le modèle chargé
        features = sample_dataset.iloc[0].to_dict()
        result = predictor.predict_match(features)

        assert result is not None
        assert result["prediction"] in ["1", "N", "2"]
        total = result["prob_1"] + result["prob_n"] + result["prob_2"]
        assert abs(total - 1.0) < 0.001

    def test_predictor_empty_batch(self):
        from models.predictor import Predictor

        predictor = Predictor(model_path="/nonexistent/path.joblib")
        df = predictor.predict_batch([])

        assert len(df) == 0

    def test_all_strategies_valid(self, sample_features):
        from models.predictor import Predictor

        for strategy in ["prudente", "equilibree", "audacieuse"]:
            predictor = Predictor(
                model_path="/nonexistent/path.joblib",
                strategy=strategy,
            )
            result = predictor.predict_match(sample_features)
            assert result is not None
            assert result["prediction"] in ["1", "N", "2"]
