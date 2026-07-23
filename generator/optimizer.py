"""Optimisation avancée des grilles selon le budget et la stratégie."""

from itertools import product
from functools import reduce


def optimize_grids(predictions: list, budget: int,
                   strategy: str = "equilibree") -> list:
    """Sélectionne le meilleur ensemble de grilles pour un budget donné.

    Stratégie prudente : peu de variantes, focus sur les favoris
    Stratégie équilibrée : variantes modérées sur les matchs incertains
    Stratégie audacieuse : plus de variantes, inclut des surprises

    Args:
        predictions: liste de dicts avec prob_1, prob_n, prob_2,
                     prediction, confiance, probas
        budget: nombre max de grilles
        strategy: 'prudente', 'equilibree', 'audacieuse'

    Returns:
        liste de dicts {resultats, confiance, probabilite, matchs}
    """
    n_matchs = len(predictions)
    base_results = [p["prediction"] for p in predictions]

    # Indices triés par confiance croissante
    sorted_indices = sorted(
        range(n_matchs), key=lambda i: predictions[i]["confiance"]
    )

    # Nombre de matchs à varier selon la stratégie
    if strategy == "prudente":
        k = min(2, n_matchs)
    elif strategy == "audacieuse":
        k = min(5, n_matchs)
    else:  # equilibree
        k = min(3, n_matchs)

    vary_indices = sorted_indices[:k]

    # Pour chaque match variable, les résultats possibles
    options_per_match = []
    for idx in vary_indices:
        pred = predictions[idx]
        probas = pred["probas"]
        sorted_results = sorted(probas.items(), key=lambda x: x[1], reverse=True)

        if strategy == "prudente":
            # Seulement le favori + le 2e choix
            options = [r for r, _ in sorted_results[:2]]
        elif strategy == "audacieuse":
            # Tous les résultats possibles
            options = [r for r, _ in sorted_results]
        else:
            # Favori + 2e choix, parfois le 3e si la confiance est basse
            if pred["confiance"] < 0.10:
                options = [r for r, _ in sorted_results]
            else:
                options = [r for r, _ in sorted_results[:2]]

        options_per_match.append(options)

    # Générer toutes les combinaisons
    all_grids = []
    seen = set()

    for combo in product(*options_per_match):
        variant_results = list(base_results)
        for i, alt in enumerate(combo):
            variant_results[vary_indices[i]] = alt

        resultats = "".join(variant_results)
        if resultats in seen:
            continue
        seen.add(resultats)

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
        # Confiance = moyenne des probas du résultat choisi pour chaque match
        confiance = sum(
            pred["probas"][variant_results[j]]
            for j, pred in enumerate(predictions)
        ) / n_matchs

        all_grids.append({
            "resultats": resultats,
            "confiance": confiance,
            "probabilite": prob,
            "matchs": matchs_detail,
        })

    # Trier par probabilité décroissante et limiter au budget
    all_grids.sort(key=lambda g: g["probabilite"], reverse=True)
    return all_grids[:budget]


def compute_grid_probability(predictions_or_grid: list,
                             resultats: str = None) -> float:
    """Calcule la probabilité combinée d'une grille (produit des probas).

    Args:
        predictions_or_grid: liste de dicts avec probas ou prob_1/prob_n/prob_2
        resultats: chaîne de résultats (ex: "1N21121"). Si None, utilise
                   la prédiction favorite de chaque match.

    Returns:
        float: probabilité combinée (produit)
    """
    prob = 1.0

    for i, pred in enumerate(predictions_or_grid):
        if resultats and i < len(resultats):
            res = resultats[i]
        else:
            res = pred.get("prediction", "1")

        # Accès aux probas via le dict 'probas' ou les clés directes
        probas = pred.get("probas")
        if probas:
            p = probas.get(res, 1 / 3)
        else:
            key = {"1": "prob_1", "N": "prob_n", "2": "prob_2"}.get(res, "prob_1")
            p = pred.get(key, 1 / 3)

        prob *= p

    return prob


def compute_grid_expected_value(grid: list, rapport_moyen: float) -> float:
    """Calcule l'espérance de gain d'une grille.

    Args:
        grid: liste de dicts avec les détails par match
        rapport_moyen: rapport moyen en euros pour le type de grille

    Returns:
        float: espérance de gain (probabilité * rapport - mise)
    """
    # Calculer la probabilité combinée depuis les matchs de la grille
    prob = 1.0
    for match in grid:
        prediction = match.get("prediction", "1")
        key = {"1": "prob_1", "N": "prob_n", "2": "prob_2"}.get(prediction, "prob_1")
        p = match.get(key, 1 / 3)
        prob *= p

    # Espérance = proba * rapport - mise (1€ par grille)
    return prob * rapport_moyen - 1.0
