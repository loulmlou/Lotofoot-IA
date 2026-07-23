"""Prédicteur de matchs — interface haut-niveau pour la prédiction de résultats."""

import os

import numpy as np
import pandas as pd

from analysis.baseline import LABEL_NAMES
from models.scoring import compute_final_score
from models.train import DEFAULT_MODEL_DIR, load_model


# Seuils de confiance par stratégie
STRATEGY_THRESHOLDS = {
    "prudente": 0.20,      # ne garde que les matchs avec haute confiance
    "equilibree": 0.10,    # seuil modéré
    "audacieuse": 0.0,     # accepte tout, y compris les surprises
}


class Predictor:
    """Interface haut-niveau pour prédire les résultats de matchs.

    Charge un modèle entraîné et fournit des méthodes de prédiction
    avec gestion des stratégies (prudente, equilibree, audacieuse).
    """

    def __init__(self, model_path=None, strategy="equilibree"):
        """Initialise le prédicteur.

        Args:
            model_path: chemin vers le fichier .joblib du modèle.
                        Si None, cherche le meilleur modèle dans data/models/
            strategy: stratégie de prédiction ('prudente', 'equilibree', 'audacieuse')
        """
        self.strategy = strategy
        self.confidence_threshold = STRATEGY_THRESHOLDS.get(strategy, 0.10)

        self.model = None
        self.preprocessor = None
        self.feature_cols = None

        self._load(model_path)

    def _load(self, model_path=None):
        """Charge le modèle depuis le disque."""
        if model_path is None:
            # Chercher le meilleur modèle disponible
            for name in ["xgboost", "lightgbm"]:
                candidate = os.path.join(DEFAULT_MODEL_DIR, f"{name}.joblib")
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        if model_path is None or not os.path.exists(model_path):
            # Pas de modèle disponible — le scoring fonctionnera sans le composant IA
            return

        artifact = load_model(model_path)
        self.model = artifact["model"]
        self.preprocessor = artifact["preprocessor"]
        self.feature_cols = artifact["feature_cols"]

    def predict_match(self, match_or_features, grid_type=None, weights=None):
        """Prédit le résultat d'un match.

        Args:
            match_or_features: dict de features du match
                               (ou objet Match ORM à convertir)
            grid_type: type de grille Loto Foot (optionnel)
            weights: poids de scoring personnalisés (optionnel)

        Returns:
            dict {
                prob_1, prob_n, prob_2: probabilités,
                prediction: '1'/'N'/'2',
                confiance: float,
                detail_scores: dict,
                filtre: bool (True si le match passe le filtre de la stratégie),
            }
            ou None si les features ne peuvent pas être calculées
        """
        # Si c'est un objet ORM Match, calculer les features
        if hasattr(match_or_features, "equipe_dom_id"):
            from analysis.features import build_match_features
            from database.connection import SessionLocal
            session = SessionLocal()
            try:
                features = build_match_features(match_or_features, session=session)
            finally:
                session.close()
        else:
            features = match_or_features

        if not features:
            return None

        result = compute_final_score(
            features,
            model=self.model,
            preprocessor=self.preprocessor,
            feature_cols=self.feature_cols,
            weights=weights,
            grid_type=grid_type,
        )

        # Appliquer le filtre de stratégie
        result["filtre"] = result["confiance"] >= self.confidence_threshold

        # En mode audacieuse, on cherche aussi les surprises potentielles
        if self.strategy == "audacieuse":
            cote_surprise = features.get("cote_surprise", 0) or 0
            if cote_surprise > 0.4:
                result["surprise_potentielle"] = True

        return result

    def predict_batch(self, matches, grid_type=None, weights=None):
        """Prédit en batch pour une liste de matchs.

        Args:
            matches: liste de dicts de features ou d'objets Match
            grid_type: type de grille Loto Foot
            weights: poids personnalisés

        Returns:
            pd.DataFrame avec colonnes prob_1, prob_n, prob_2, prediction,
            confiance, filtre
        """
        results = []

        for match in matches:
            pred = self.predict_match(match, grid_type=grid_type, weights=weights)
            if pred is not None:
                # Ajouter des infos de contexte si disponibles
                row = {
                    "prob_1": pred["prob_1"],
                    "prob_n": pred["prob_n"],
                    "prob_2": pred["prob_2"],
                    "prediction": pred["prediction"],
                    "confiance": pred["confiance"],
                    "filtre": pred["filtre"],
                }

                # Ajouter les identifiants si c'est un dict de features
                if isinstance(match, dict):
                    for key in ["match_id", "equipe_dom_id", "equipe_ext_id",
                                "competition_id", "date"]:
                        if key in match:
                            row[key] = match[key]

                results.append(row)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def predict_from_ids(self, equipe_dom_id, equipe_ext_id,
                         competition_id, date, session=None):
        """Prédit le résultat d'un match à partir des IDs des équipes.

        Utile pour prédire des matchs à venir pas encore en base.

        Args:
            equipe_dom_id: ID de l'équipe à domicile
            equipe_ext_id: ID de l'équipe à l'extérieur
            competition_id: ID de la compétition
            date: date du match (datetime ou str)
            session: session SQLAlchemy (optionnel)

        Returns:
            dict de prédiction (même format que predict_match)
        """
        from datetime import datetime as dt
        from analysis.features import (
            compute_form, compute_h2h, compute_standing,
            compute_goal_stats, compute_context,
        )
        from database.connection import SessionLocal

        if isinstance(date, str):
            date = pd.to_datetime(date)

        close = session is None
        if close:
            session = SessionLocal()

        try:
            # Calculer chaque groupe de features
            dom_forme = compute_form(equipe_dom_id, date, n=5, session=session)
            ext_forme = compute_form(equipe_ext_id, date, n=5, session=session)
            h2h = compute_h2h(equipe_dom_id, equipe_ext_id, date, n=5, session=session)
            dom_standing = compute_standing(equipe_dom_id, competition_id, date, session=session)
            ext_standing = compute_standing(equipe_ext_id, competition_id, date, session=session)
            dom_goals = compute_goal_stats(equipe_dom_id, date, n=10, session=session)
            ext_goals = compute_goal_stats(equipe_ext_id, date, n=10, session=session)
            dom_ctx = compute_context(equipe_dom_id, date, competition_id, session=session)

            features = {
                "dom_forme_pts": dom_forme["points"],
                "dom_forme_ratio_vic": dom_forme["ratio_victoires"],
                "dom_forme_buts_m": dom_forme["buts_marques"],
                "dom_forme_buts_e": dom_forme["buts_encaisses"],
                "dom_forme_nb": dom_forme["nb_matchs"],
                "ext_forme_pts": ext_forme["points"],
                "ext_forme_ratio_vic": ext_forme["ratio_victoires"],
                "ext_forme_buts_m": ext_forme["buts_marques"],
                "ext_forme_buts_e": ext_forme["buts_encaisses"],
                "ext_forme_nb": ext_forme["nb_matchs"],
                "h2h_nb": h2h["nb_matchs"],
                "h2h_pct_dom": h2h["pct_dom"],
                "h2h_pct_nul": h2h["pct_nul"],
                "h2h_pct_ext": h2h["pct_ext"],
                "dom_position": dom_standing.get("position"),
                "ext_position": ext_standing.get("position"),
                "diff_position": (
                    (dom_standing["position"] - ext_standing["position"])
                    if dom_standing.get("position") and ext_standing.get("position")
                    else None
                ),
                "dom_classement_pts": dom_standing.get("points", 0),
                "ext_classement_pts": ext_standing.get("points", 0),
                "dom_moy_buts_m": dom_goals["moy_buts_marques"],
                "dom_moy_buts_e": dom_goals["moy_buts_encaisses"],
                "dom_over25": dom_goals["over_2_5_ratio"],
                "ext_moy_buts_m": ext_goals["moy_buts_marques"],
                "ext_moy_buts_e": ext_goals["moy_buts_encaisses"],
                "ext_over25": ext_goals["over_2_5_ratio"],
                "dom_jours_repos": dom_ctx["jours_repos"],
                "ext_jours_repos": compute_context(
                    equipe_ext_id, date, competition_id, session=session
                )["jours_repos"],
                "avantage_dom_ligue": dom_ctx["avantage_dom_ligue"],
                # Pas de cotes pour un match à venir sans données bookmaker
                "prob_1": None,
                "prob_n": None,
                "prob_2": None,
                "cote_1": None,
                "cote_n": None,
                "cote_2": None,
                "cote_surprise": None,
            }

            return self.predict_match(features)

        finally:
            if close:
                session.close()
