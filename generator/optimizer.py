"""Optimisation avancée des grilles selon le budget et la stratégie."""

from itertools import product
from functools import reduce

from loguru import logger


def _compute_grid_profile(resultats: str) -> str:
    """Calcule le profil 1N2 d'une grille (ex: '1N21121' -> '4-1-2')."""
    nb1 = resultats.count("1")
    nb_n = resultats.count("N")
    nb2 = resultats.count("2")
    return f"{nb1}-{nb_n}-{nb2}"


def _select_diverse_grids(grids: list, budget: int) -> list:
    """Selectionne les grilles en diversifiant les profils 1N2.

    Prend la meilleure grille de chaque profil distinct, puis complete
    avec les meilleures restantes si le budget le permet.
    Cela evite de soumettre 5 grilles identiques genre 5-1-1, 5-1-1, 5-1-1...

    Args:
        grids: liste de grilles triees par score decroissant
        budget: nombre de grilles a selectionner

    Returns:
        liste de grilles selectionnees (au plus budget)
    """
    if len(grids) <= budget:
        return grids

    selected = []
    seen_profiles = set()

    # Phase 1 : une grille par profil distinct (dans l'ordre du score)
    for g in grids:
        profil = g.get("profil") or _compute_grid_profile(g["resultats"])
        if profil not in seen_profiles:
            selected.append(g)
            seen_profiles.add(profil)
            if len(selected) >= budget:
                return selected

    # Phase 2 : completer avec les meilleures grilles restantes
    for g in grids:
        if g not in selected:
            selected.append(g)
            if len(selected) >= budget:
                break

    return selected


def optimize_grids(predictions: list, budget: int,
                   strategy: str = "equilibree",
                   grid_type: str = None,
                   grid_metrics: dict = None) -> list:
    """Sélectionne le meilleur ensemble de grilles pour un budget donné.

    Stratégie prudente : peu de variantes, focus sur les favoris
    Stratégie équilibrée : variantes modérées sur les matchs incertains
    Stratégie audacieuse : plus de variantes, inclut des surprises

    Args:
        predictions: liste de dicts avec prob_1, prob_n, prob_2,
                     prediction, confiance, probas
        budget: nombre max de grilles
        strategy: 'prudente', 'equilibree', 'audacieuse'
        grid_type: type de grille (LF7, LF8, LF12, LF15) pour
                   pondérer par les stats historiques de combinaisons 1N2
        grid_metrics: métriques de la grille (difficulty, moyenne_cote_fav)
                      pour ajuster le nombre de matchs variés

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

    # Ajuster k selon les indicateurs de la grille
    if grid_metrics:
        difficulty = grid_metrics.get("difficulty")
        inv_spread = grid_metrics.get("inv_spread_sum")
        std_fav = grid_metrics.get("std_cote_fav")
        nb_serres = grid_metrics.get("nb_matchs_serres", 0)

        # inv_spread_sum < 1.98 => grille "facile" (1 surprise en moy)
        # inv_spread_sum > 2.5 => grille "difficile" (3 surprises en moy)
        if inv_spread is not None:
            if inv_spread < 2.0 and strategy != "audacieuse":
                k = max(k - 1, 1)  # Peu de surprises attendues
            elif inv_spread > 3.0 and strategy != "prudente":
                k = min(k + 1, n_matchs)  # Beaucoup de surprises attendues

        # std_cote_fav < 0.26 => cotes homogenes, peu de surprises
        elif std_fav is not None:
            if std_fav < 0.26 and strategy != "audacieuse":
                k = max(k - 1, 1)
            elif std_fav > 0.45 and strategy != "prudente":
                k = min(k + 1, n_matchs)

        # Fallback sur la difficulte Pronosoft
        elif difficulty is not None:
            if difficulty > 8.0 and strategy != "prudente":
                k = min(k + 1, n_matchs)
            elif difficulty < 4.0 and strategy != "audacieuse":
                k = max(k - 1, 1)

        # Si beaucoup de matchs serres, augmenter la couverture
        if nb_serres and nb_serres >= 3:
            k = min(k + 1, n_matchs)

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
            # Seuil releve a 0.15 pour inclure plus de "N" sur les matchs intermediaires
            if pred["confiance"] < 0.15:
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

    # Pondérer par la fréquence historique du profil 1N2
    combinaisons_stats = {}
    if grid_type:
        try:
            from collectors.pronosoft_scraper import fetch_combinaisons_stats
            combinaisons_stats = fetch_combinaisons_stats(grid_type)
        except Exception as e:
            logger.warning(f"Impossible de charger les stats combinaisons: {e}")

    if combinaisons_stats:
        # Fréquence plancher pour les profils absents des stats
        freq_plancher = min(combinaisons_stats.values()) * 0.1
        for grid in all_grids:
            profil = _compute_grid_profile(grid["resultats"])
            profil_weight = combinaisons_stats.get(profil, freq_plancher)
            grid["profil"] = profil
            grid["profil_weight"] = profil_weight
            grid["score"] = grid["probabilite"] * profil_weight

        all_grids.sort(key=lambda g: g["score"], reverse=True)
    else:
        all_grids.sort(key=lambda g: g["probabilite"], reverse=True)

    # --- Diversification des profils 1N2 ---
    # Eviter de soumettre N grilles avec le meme profil.
    # Selectionner le budget en favorisant la diversite des profils.
    selected = _select_diverse_grids(all_grids, budget)
    return selected


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
