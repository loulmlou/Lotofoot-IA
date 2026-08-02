"""Optimisation des pondérations du modèle OddsPredictor par backtesting.

Utilise l'historique des grilles LF7 scrapé depuis Pronosoft pour trouver
les poids optimaux des signaux (cotes, consensus, cyborg, difficulte, historique).
"""

import json
import os

import numpy as np
from loguru import logger

from collectors.pronosoft_history import load_history, DEFAULT_HISTORY_PATH
from models.odds_predictor import OddsPredictor, DEFAULT_WEIGHTS


class WeightOptimizer:
    """Optimise les poids du modèle OddsPredictor sur l'historique LF7."""

    def __init__(self, history_path: str = DEFAULT_HISTORY_PATH):
        """Initialise avec le chemin de l'historique.

        Args:
            history_path: chemin vers le fichier JSON d'historique
        """
        self.history_path = history_path
        self._grids = None

    @property
    def grids(self) -> list:
        """Charge l'historique (lazy loading)."""
        if self._grids is None:
            self._grids = load_history(self.history_path)
        return self._grids

    def evaluate_weights(self, weights: dict,
                         grids: list = None,
                         metric: str = "correct_6_plus") -> float:
        """Évalue un jeu de poids sur un ensemble de grilles.

        Args:
            weights: dict des poids (cotes, consensus, cyborg, difficulte, historique)
            grids: liste de grilles à évaluer (None = toutes)
            metric: métrique à maximiser
                - "correct_7": nombre de grilles 7/7
                - "correct_6_plus": nombre de grilles >= 6/7
                - "accuracy": % de matchs correctement prédits
                - "roi": retour sur investissement simulé

        Returns:
            float: valeur de la métrique (plus élevé = meilleur)
        """
        if grids is None:
            grids = self.grids

        if not grids:
            return 0.0

        predictor = OddsPredictor(weights=weights)

        total_correct = 0
        total_matchs = 0
        grids_7_sur_7 = 0
        grids_6_plus = 0
        total_roi = 0.0
        total_grids = 0

        for grid in grids:
            matches = grid.get("matches", [])
            resultats = grid.get("resultats", "")
            if not matches or not resultats or len(resultats) != len(matches):
                continue

            total_grids += 1
            predictions = predictor.predict_grid(grid)
            grid_correct = 0

            for i, pred in enumerate(predictions):
                if i < len(resultats):
                    actual = resultats[i]
                    if pred["prediction"] == actual:
                        grid_correct += 1
                        total_correct += 1
                    total_matchs += 1

            if grid_correct == 7:
                grids_7_sur_7 += 1
            if grid_correct >= 6:
                grids_6_plus += 1

            # ROI: rapport si 7/7
            if grid_correct == 7:
                rapports = grid.get("rapports", {})
                r7 = rapports.get("7_sur_7", {})
                montant = r7.get("montant", 0)
                total_roi += montant - 1.0
            else:
                total_roi -= 1.0

        if metric == "correct_7":
            return float(grids_7_sur_7)
        elif metric == "correct_6_plus":
            return float(grids_6_plus)
        elif metric == "accuracy":
            return total_correct / total_matchs if total_matchs > 0 else 0.0
        elif metric == "roi":
            return total_roi / total_grids if total_grids > 0 else 0.0
        else:
            return float(grids_6_plus)

    def optimize_global(self, metric: str = "correct_6_plus",
                        n_restarts: int = 10) -> dict:
        """Optimise les poids globalement sur toutes les grilles.

        Args:
            metric: métrique à maximiser
            n_restarts: nombre de restarts aléatoires

        Returns:
            dict avec 'weights' (meilleurs poids), 'score' (meilleure valeur)
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.warning("scipy non disponible, utilisation de la recherche par grille")
            return self._grid_search(metric)

        weight_names = list(DEFAULT_WEIGHTS.keys())
        n_weights = len(weight_names)

        best_score = -np.inf
        best_weights = DEFAULT_WEIGHTS.copy()

        def objective(x):
            # x est un vecteur de poids bruts, on normalise par softmax
            exp_x = np.exp(x - np.max(x))
            w = exp_x / exp_x.sum()
            weights = {name: float(w[i]) for i, name in enumerate(weight_names)}
            score = self.evaluate_weights(weights, metric=metric)
            return -score  # minimize

        for restart in range(n_restarts):
            if restart == 0:
                # Premier essai: poids par défaut
                x0 = np.array([DEFAULT_WEIGHTS[n] for n in weight_names])
                x0 = np.log(x0 + 1e-8)
            else:
                x0 = np.random.randn(n_weights) * 0.5

            try:
                result = minimize(
                    objective, x0,
                    method="Nelder-Mead",
                    options={"maxiter": 200, "xatol": 0.01, "fatol": 0.1},
                )

                score = -result.fun
                if score > best_score:
                    best_score = score
                    exp_x = np.exp(result.x - np.max(result.x))
                    w = exp_x / exp_x.sum()
                    best_weights = {
                        name: round(float(w[i]), 4)
                        for i, name in enumerate(weight_names)
                    }
            except Exception as e:
                logger.warning(f"Restart {restart} échoué: {e}")
                continue

        logger.info(f"Optimisation globale ({metric}): score={best_score:.4f}")
        logger.info(f"Poids optimaux: {best_weights}")

        return {"weights": best_weights, "score": best_score}

    def optimize_sliding_window(self, window_size: int = 20,
                                metric: str = "accuracy") -> list[dict]:
        """Optimise les poids par fenêtre glissante.

        Pour chaque grille N, optimise sur les grilles [N-window, N-1].

        Args:
            window_size: taille de la fenêtre
            metric: métrique à optimiser

        Returns:
            liste de dicts {grid_number, weights, score}
        """
        grids = self.grids
        if len(grids) <= window_size:
            logger.warning(f"Pas assez de grilles ({len(grids)}) pour fenêtre de {window_size}")
            return []

        results = []
        for i in range(window_size, len(grids)):
            window = grids[i - window_size:i]

            # Simple grid search on the window
            best_score = -np.inf
            best_weights = DEFAULT_WEIGHTS.copy()

            for weights in self._generate_weight_candidates():
                score = self.evaluate_weights(weights, grids=window, metric=metric)
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()

            results.append({
                "grid_number": grids[i].get("grid_number", i + 1),
                "weights": best_weights,
                "score": best_score,
            })

        return results

    def get_current_weights(self, window_size: int = 20) -> dict:
        """Poids optimisés sur les N dernières grilles.

        Args:
            window_size: nombre de grilles récentes à utiliser

        Returns:
            dict des poids optimisés
        """
        grids = self.grids
        if not grids:
            return DEFAULT_WEIGHTS.copy()

        recent = grids[-window_size:]
        best_score = -np.inf
        best_weights = DEFAULT_WEIGHTS.copy()

        for weights in self._generate_weight_candidates():
            score = self.evaluate_weights(weights, grids=recent, metric="accuracy")
            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        return best_weights

    def _generate_weight_candidates(self) -> list[dict]:
        """Génère des candidats de poids pour la recherche par grille."""
        names = list(DEFAULT_WEIGHTS.keys())
        candidates = [DEFAULT_WEIGHTS.copy()]

        # Variations autour des poids par défaut
        base = [DEFAULT_WEIGHTS[n] for n in names]
        for i in range(len(names)):
            for delta in [-0.10, -0.05, 0.05, 0.10]:
                variant = list(base)
                variant[i] = max(0.0, variant[i] + delta)
                total = sum(variant)
                if total > 0:
                    variant = [v / total for v in variant]
                    candidates.append({n: round(v, 4) for n, v in zip(names, variant)})

        # Quelques profils extrêmes
        extreme_profiles = [
            {"cotes": 0.80, "consensus": 0.05, "cyborg": 0.05, "difficulte": 0.05, "historique": 0.05},
            {"cotes": 0.50, "consensus": 0.25, "cyborg": 0.10, "difficulte": 0.10, "historique": 0.05},
            {"cotes": 0.40, "consensus": 0.20, "cyborg": 0.20, "difficulte": 0.10, "historique": 0.10},
            {"cotes": 0.70, "consensus": 0.10, "cyborg": 0.10, "difficulte": 0.05, "historique": 0.05},
        ]
        candidates.extend(extreme_profiles)

        return candidates

    def _grid_search(self, metric: str) -> dict:
        """Recherche par grille (fallback sans scipy)."""
        best_score = -np.inf
        best_weights = DEFAULT_WEIGHTS.copy()

        for weights in self._generate_weight_candidates():
            score = self.evaluate_weights(weights, metric=metric)
            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        return {"weights": best_weights, "score": best_score}
