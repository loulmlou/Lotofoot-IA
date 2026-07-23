"""Entraînement des modèles ML pour la prédiction 1/N/2."""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.baseline import LABEL_MAP, LABEL_NAMES, evaluate_predictions
from analysis.dataset_builder import build_dataset, load_dataset, split_by_date


# Features utilisées pour l'entraînement (exclut les métadonnées)
FEATURE_COLS = [
    "dom_forme_pts", "dom_forme_ratio_vic", "dom_forme_buts_m", "dom_forme_buts_e", "dom_forme_nb",
    "ext_forme_pts", "ext_forme_ratio_vic", "ext_forme_buts_m", "ext_forme_buts_e", "ext_forme_nb",
    "h2h_nb", "h2h_pct_dom", "h2h_pct_nul", "h2h_pct_ext",
    "dom_position", "ext_position", "diff_position",
    "dom_classement_pts", "ext_classement_pts",
    "dom_moy_buts_m", "dom_moy_buts_e", "dom_over25",
    "ext_moy_buts_m", "ext_moy_buts_e", "ext_over25",
    "prob_1", "prob_n", "prob_2", "cote_surprise",
    "cote_1", "cote_n", "cote_2",
    "dom_jours_repos", "ext_jours_repos",
    "avantage_dom_ligue",
]

DEFAULT_MODEL_DIR = os.path.join("data", "models")


def preprocess_features(df, feature_cols=None):
    """Prétraite les features : imputation des NaN (médiane) + standardisation.

    Args:
        df: DataFrame avec les features et 'resultat'
        feature_cols: liste des colonnes features (défaut: FEATURE_COLS)

    Returns:
        tuple (X, y, feature_names, preprocessor)
            - X: array numpy des features transformées
            - y: array numpy des labels encodés (0=1, 1=N, 2=2)
            - feature_names: liste des noms de features
            - preprocessor: pipeline sklearn (fitted)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Ne garder que les colonnes présentes dans le DataFrame
    available = [c for c in feature_cols if c in df.columns]

    X_raw = df[available].copy()
    y = df["resultat"].map(LABEL_MAP).values

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X = preprocessor.fit_transform(X_raw)

    return X, y, available, preprocessor


def _apply_preprocessor(df, preprocessor, feature_cols):
    """Applique un preprocessor déjà fitted sur un DataFrame."""
    available = [c for c in feature_cols if c in df.columns]
    X_raw = df[available].copy()
    y = df["resultat"].map(LABEL_MAP).values
    X = preprocessor.transform(X_raw)
    return X, y


def train_xgboost(X_train, y_train, X_val, y_val, search_params=True):
    """Entraîne un modèle XGBoost multiclass avec optionnel hyperparameter search.

    Args:
        X_train: features d'entraînement
        y_train: labels d'entraînement
        X_val: features de validation
        y_val: labels de validation
        search_params: si True, fait un RandomizedSearchCV

    Returns:
        modèle XGBoost entraîné
    """
    from xgboost import XGBClassifier

    base_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
                "random_state": 42,
        "n_jobs": -1,
    }

    if search_params:
        param_distributions = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [1.0, 1.5, 2.0],
        }

        model = XGBClassifier(**base_params)

        cv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=20,
            cv=cv,
            scoring="neg_log_loss",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"  XGBoost best params: {search.best_params_}")
        print(f"  XGBoost best CV log-loss: {-search.best_score_:.4f}")
    else:
        best_model = XGBClassifier(
            **base_params,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    return best_model


def train_lightgbm(X_train, y_train, X_val, y_val, search_params=True):
    """Entraîne un modèle LightGBM multiclass.

    Args:
        X_train: features d'entraînement
        y_train: labels d'entraînement
        X_val: features de validation
        y_val: labels de validation
        search_params: si True, fait un RandomizedSearchCV

    Returns:
        modèle LightGBM entraîné
    """
    from lightgbm import LGBMClassifier

    base_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    if search_params:
        param_distributions = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [5, 10, 20, 50],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0, 0.01, 0.1, 1.0],
            "num_leaves": [15, 31, 63, 127],
        }

        model = LGBMClassifier(**base_params)

        cv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=20,
            cv=cv,
            scoring="neg_log_loss",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"  LightGBM best params: {search.best_params_}")
        print(f"  LightGBM best CV log-loss: {-search.best_score_:.4f}")
    else:
        best_model = LGBMClassifier(
            **base_params,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            num_leaves=31,
        )
        best_model.fit(
            X_train, y_train,
            eval_X=X_val, eval_y=y_val,
        )

    return best_model


def evaluate_model(model, X_test, y_test, cotes=None):
    """Évalue un modèle sur un jeu de test.

    Args:
        model: modèle sklearn-compatible avec predict et predict_proba
        X_test: features de test
        y_test: labels encodés (0/1/2)
        cotes: DataFrame ou list de dict avec cote_1, cote_n, cote_2

    Returns:
        dict avec accuracy, log_loss, roi, etc.
    """
    y_pred_encoded = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Reconvertir en labels textuels
    y_true_labels = [LABEL_NAMES[i] for i in y_test]
    y_pred_labels = [LABEL_NAMES[i] for i in y_pred_encoded]

    result = evaluate_predictions(
        y_true_labels, y_pred_labels,
        y_proba=y_proba, cotes=cotes,
    )
    return result


def save_model(model, preprocessor, feature_cols, path=None, name="xgboost"):
    """Sauvegarde un modèle entraîné et son preprocessor.

    Args:
        model: modèle entraîné
        preprocessor: pipeline de prétraitement
        feature_cols: colonnes de features utilisées
        path: répertoire de destination (défaut: data/models/)
        name: nom du modèle (pour le fichier)
    """
    if path is None:
        path = DEFAULT_MODEL_DIR

    os.makedirs(path, exist_ok=True)

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "trained_at": datetime.now().isoformat(),
    }

    filepath = os.path.join(path, f"{name}.joblib")
    joblib.dump(artifact, filepath)
    print(f"  Modèle sauvegardé: {filepath}")
    return filepath


def load_model(path):
    """Charge un modèle et son preprocessor depuis le disque.

    Args:
        path: chemin vers le fichier .joblib

    Returns:
        dict avec model, preprocessor, feature_cols, trained_at
    """
    artifact = joblib.load(path)
    return artifact


def train_all(df=None, search_params=True):
    """Pipeline complet d'entraînement : load data, preprocess, train, evaluate, save.

    Args:
        df: DataFrame (si None, construit le dataset depuis la BDD)
        search_params: si True, optimise les hyperparamètres (plus lent)

    Returns:
        dict avec les résultats d'évaluation pour chaque modèle
    """
    print("=" * 60)
    print("ENTRAÎNEMENT DES MODÈLES ML")
    print("=" * 60)

    # 1. Charger le dataset
    if df is None:
        dataset_path = os.path.join("data", "dataset.csv")
        if os.path.exists(dataset_path):
            print(f"\nChargement du dataset depuis {dataset_path}...")
            df = load_dataset(dataset_path)
        else:
            print("\nConstruction du dataset depuis la base de données...")
            df = build_dataset()
            save_path = os.path.join("data", "dataset.csv")
            from analysis.dataset_builder import save_dataset
            save_dataset(df, save_path)
            print(f"  Dataset sauvegardé: {save_path}")

    print(f"  Dataset: {df.shape[0]} matchs × {df.shape[1]} colonnes")

    # 2. Split walk-forward
    print("\nSplit walk-forward par date...")
    train_df, val_df, test_df = split_by_date(df, "2024-01-01", "2025-01-01")
    print(f"  Train: {len(train_df)} matchs (< 2024-01-01)")
    print(f"  Validation: {len(val_df)} matchs (2024-01-01 à 2025-01-01)")
    print(f"  Test: {len(test_df)} matchs (> 2025-01-01)")

    if len(train_df) < 100:
        print("ATTENTION: Pas assez de données d'entraînement. Ajustement des dates...")
        # Fallback: utiliser des percentiles
        dates = df["date"].sort_values()
        n = len(dates)
        train_end = str(dates.iloc[int(n * 0.6)].date())
        val_end = str(dates.iloc[int(n * 0.8)].date())
        train_df, val_df, test_df = split_by_date(df, train_end, val_end)
        print(f"  Train: {len(train_df)} matchs (< {train_end})")
        print(f"  Validation: {len(val_df)} matchs ({train_end} à {val_end})")
        print(f"  Test: {len(test_df)} matchs (> {val_end})")

    # 3. Prétraitement
    print("\nPrétraitement des features...")
    X_train, y_train, feature_cols, preprocessor = preprocess_features(train_df)
    X_val, y_val = _apply_preprocessor(val_df, preprocessor, feature_cols)
    X_test, y_test = _apply_preprocessor(test_df, preprocessor, feature_cols)

    print(f"  Features: {len(feature_cols)} colonnes")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # Vérifier pas de NaN
    assert not np.isnan(X_train).any(), "NaN détectés après prétraitement (train)"
    assert not np.isnan(X_val).any(), "NaN détectés après prétraitement (val)"
    assert not np.isnan(X_test).any(), "NaN détectés après prétraitement (test)"

    # Cotes pour calcul ROI
    cotes_test = None
    if "cote_1" in test_df.columns:
        cotes_test = test_df[["cote_1", "cote_n", "cote_2"]].to_dict("records")

    results = {}

    # 4. XGBoost
    print("\n--- XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, search_params=search_params)
    xgb_eval = evaluate_model(xgb_model, X_test, y_test, cotes=cotes_test)
    results["xgboost"] = xgb_eval
    print(f"  Accuracy: {xgb_eval['accuracy']:.4f}")
    print(f"  Log-loss: {xgb_eval.get('log_loss', 'N/A')}")
    if "roi" in xgb_eval:
        print(f"  ROI: {xgb_eval['roi']:.4f}")

    save_model(xgb_model, preprocessor, feature_cols, name="xgboost")

    # 5. LightGBM
    print("\n--- LightGBM ---")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, search_params=search_params)
    lgb_eval = evaluate_model(lgb_model, X_test, y_test, cotes=cotes_test)
    results["lightgbm"] = lgb_eval
    print(f"  Accuracy: {lgb_eval['accuracy']:.4f}")
    print(f"  Log-loss: {lgb_eval.get('log_loss', 'N/A')}")
    if "roi" in lgb_eval:
        print(f"  ROI: {lgb_eval['roi']:.4f}")

    save_model(lgb_model, preprocessor, feature_cols, name="lightgbm")

    # 6. Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"  XGBoost  — Accuracy: {xgb_eval['accuracy']:.4f}, "
          f"Log-loss: {xgb_eval.get('log_loss', 'N/A')}")
    print(f"  LightGBM — Accuracy: {lgb_eval['accuracy']:.4f}, "
          f"Log-loss: {lgb_eval.get('log_loss', 'N/A')}")

    # Sélectionner le meilleur modèle
    if xgb_eval.get("log_loss", 999) <= lgb_eval.get("log_loss", 999):
        best_name = "xgboost"
    else:
        best_name = "lightgbm"
    results["best"] = best_name
    print(f"  Meilleur modèle: {best_name}")

    return results
