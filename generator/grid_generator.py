"""Générateur de grilles Loto Foot optimisées."""

from itertools import product

from models.odds_predictor import OddsPredictor
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

    def __init__(self, predictor: OddsPredictor = None, strategy: str = "equilibree"):
        """Initialise avec un OddsPredictor et une stratégie.

        Args:
            predictor: instance de OddsPredictor. Si None, en crée un par défaut.
            strategy: stratégie de génération ('prudente', 'equilibree', 'audacieuse')
        """
        if predictor is None:
            predictor = OddsPredictor(strategy=strategy)
        self.predictor = predictor
        self.strategy = strategy

    def generate(self, grid_data: dict, grid_type: str = "LF7",
                 budget: int = 5) -> list:
        """Génère les grilles optimisées pour un ensemble de matchs.

        Args:
            grid_data: dict avec 'matches' (liste de dicts de cotes par match)
                       et optionnellement des métriques de grille
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
        predictions = self.predictor.predict_grid(grid_data)
        if not predictions:
            return []

        # Ajouter probas dict pour compatibilité avec optimizer
        for pred in predictions:
            if "probas" not in pred:
                pred["probas"] = {
                    "1": pred["prob_1"],
                    "N": pred["prob_n"],
                    "2": pred["prob_2"],
                }

        grid_metrics = {
            "difficulty": grid_data.get("difficulty"),
            "moyenne_cote_fav": grid_data.get("moyenne_cote_fav"),
            "inv_spread_sum": grid_data.get("inv_spread_sum"),
            "std_cote_fav": grid_data.get("std_cote_fav"),
            "accord_cotes_joueurs": grid_data.get("accord_cotes_joueurs"),
            "nb_matchs_serres": grid_data.get("nb_matchs_serres"),
        }

        grids = optimize_grids(
            predictions, budget, self.strategy, grid_type,
            grid_metrics=grid_metrics,
        )
        return self._rank_grids(grids)

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
        """Variantes par remplacement des matchs les moins confiants."""
        n_matchs = len(predictions)
        base_results = [p["prediction"] for p in predictions]

        sorted_indices = sorted(
            range(n_matchs), key=lambda i: predictions[i]["confiance"]
        )

        if self.strategy == "prudente":
            k = min(1, n_matchs)
        elif self.strategy == "audacieuse":
            k = min(4, n_matchs)
        else:
            k = min(3, n_matchs)

        vary_indices = sorted_indices[:k]

        alternatives_per_match = []
        for idx in vary_indices:
            pred = predictions[idx]
            probas = pred["probas"]
            sorted_results = sorted(probas.items(), key=lambda x: x[1], reverse=True)
            alts = [r for r, _ in sorted_results if r != pred["prediction"]]
            alternatives_per_match.append(alts)

        variants = []
        for combo in product(*alternatives_per_match):
            variant_results = list(base_results)
            for i, alt in enumerate(combo):
                variant_results[vary_indices[i]] = alt

            resultats = "".join(variant_results)

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

        variants.sort(key=lambda g: g["probabilite"], reverse=True)
        return variants[:max(n_variants, 0)]

    def _rank_grids(self, grids: list) -> list:
        """Trie les grilles par score de confiance décroissant."""
        return sorted(grids, key=lambda g: g["probabilite"], reverse=True)
