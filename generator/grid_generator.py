"""Générateur de grilles Loto Foot optimisées."""

from itertools import product

from models.predictor import Predictor
from generator.optimizer import optimize_grids, compute_grid_probability


# Nombre de matchs par type de grille
GRID_SIZES = {
    "LF7": 7,
    "LF8": 8,
    "LF12": 12,
    "LF15": 15,
}

RESULTS = ["1", "N", "2"]


class GridGenerator:
    """Génère des grilles Loto Foot optimisées à partir des prédictions."""

    def __init__(self, predictor: Predictor = None, strategy: str = "equilibree"):
        """Initialise avec un Predictor et une stratégie.

        Args:
            predictor: instance de Predictor. Si None, en crée un par défaut.
            strategy: stratégie de génération ('prudente', 'equilibree', 'audacieuse')
        """
        if predictor is None:
            predictor = Predictor(strategy=strategy)
        self.predictor = predictor
        self.strategy = strategy

    def generate(self, matches: list, grid_type: str = "LF7",
                 budget: int = 5) -> list:
        """Génère les grilles optimisées pour un ensemble de matchs.

        Args:
            matches: liste de dicts de features pour chaque match
            grid_type: type de grille (LF7, LF8, LF12, LF15)
            budget: nombre max de grilles à générer (= mises)

        Returns:
            liste de dicts {
                resultats: str (ex: "1N21121"),
                confiance: float,
                probabilite: float (proba combinée),
                matchs: list[dict] (détails par match),
            }
        """
        predictions = self._predict_all(matches, grid_type)
        if not predictions:
            return []

        grids = optimize_grids(predictions, budget, self.strategy, grid_type)
        return self._rank_grids(grids)

    def _predict_all(self, matches: list, grid_type: str) -> list:
        """Prédit tous les matchs avec le Predictor.

        Returns:
            liste de dicts par match, chacun contenant:
            - prob_1, prob_n, prob_2, prediction, confiance
            - probas: dict {1: p1, N: pn, 2: p2} pour accès facile
            - features: dict des features originales du match
        """
        predictions = []
        for match in matches:
            pred = self.predictor.predict_match(match, grid_type=grid_type)
            if pred is None:
                # Fallback: distribution uniforme
                pred = {
                    "prob_1": 1 / 3,
                    "prob_n": 1 / 3,
                    "prob_2": 1 / 3,
                    "prediction": "1",
                    "confiance": 0.0,
                }

            pred["probas"] = {
                "1": pred["prob_1"],
                "N": pred["prob_n"],
                "2": pred["prob_2"],
            }
            pred["features"] = match
            predictions.append(pred)

        return predictions

    def _generate_base_grid(self, predictions: list) -> dict:
        """Grille de base = tous les favoris.

        Returns:
            dict {resultats, confiance, probabilite, matchs}
        """
        resultats = ""
        matchs_detail = []

        for pred in predictions:
            resultats += pred["prediction"]
            matchs_detail.append({
                "prediction": pred["prediction"],
                "prob_1": pred["prob_1"],
                "prob_n": pred["prob_n"],
                "prob_2": pred["prob_2"],
                "confiance": pred["confiance"],
            })

        prob = compute_grid_probability(predictions, resultats)
        confiance = sum(p["confiance"] for p in predictions) / len(predictions)

        return {
            "resultats": resultats,
            "confiance": confiance,
            "probabilite": prob,
            "matchs": matchs_detail,
        }

    def _generate_variants(self, predictions: list, n_variants: int) -> list:
        """Variantes par remplacement des matchs les moins confiants.

        Algorithme :
        - Trier les matchs par confiance croissante
        - Pour les K matchs les moins confiants, alterner entre
          le 2e et 3e résultat le plus probable
        - Combinatoire contrôlée pour rester dans le budget
        """
        n_matchs = len(predictions)
        base_results = [p["prediction"] for p in predictions]

        # Indices triés par confiance croissante (les moins confiants en premier)
        sorted_indices = sorted(
            range(n_matchs), key=lambda i: predictions[i]["confiance"]
        )

        # Nombre de matchs à varier selon la stratégie
        if self.strategy == "prudente":
            k = min(1, n_matchs)
        elif self.strategy == "audacieuse":
            k = min(4, n_matchs)
        else:  # equilibree
            k = min(3, n_matchs)

        # Indices des matchs à varier (les K moins confiants)
        vary_indices = sorted_indices[:k]

        # Pour chaque match variable, calculer les alternatives
        alternatives_per_match = []
        for idx in vary_indices:
            pred = predictions[idx]
            probas = pred["probas"]
            # Trier par proba décroissante
            sorted_results = sorted(probas.items(), key=lambda x: x[1], reverse=True)
            # Le favori est déjà dans la grille de base, prendre les alternatives
            alts = [r for r, _ in sorted_results if r != pred["prediction"]]
            alternatives_per_match.append(alts)

        # Générer les combinaisons (chaque match variable prend une alternative)
        variants = []
        for combo in product(*alternatives_per_match):
            variant_results = list(base_results)
            for i, alt in enumerate(combo):
                variant_results[vary_indices[i]] = alt

            resultats = "".join(variant_results)

            # Construire les détails
            matchs_detail = []
            for j, pred in enumerate(predictions):
                matchs_detail.append({
                    "prediction": variant_results[j],
                    "prob_1": pred["prob_1"],
                    "prob_n": pred["prob_n"],
                    "prob_2": pred["prob_2"],
                    "confiance": pred["confiance"],
                })

            prob = compute_grid_probability(predictions, resultats)
            confiance = sum(
                pred["probas"][variant_results[j]]
                for j, pred in enumerate(predictions)
            ) / n_matchs

            variants.append({
                "resultats": resultats,
                "confiance": confiance,
                "probabilite": prob,
                "matchs": matchs_detail,
            })

        # Limiter et trier
        variants.sort(key=lambda g: g["probabilite"], reverse=True)
        return variants[:max(n_variants, 0)]

    def _rank_grids(self, grids: list) -> list:
        """Trie les grilles par score de confiance décroissant."""
        return sorted(grids, key=lambda g: g["probabilite"], reverse=True)
