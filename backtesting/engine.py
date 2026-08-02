"""Moteur de backtesting — évalue les prédictions basées sur les cotes
sur l'historique des grilles scrapé depuis Pronosoft."""

import os
from collections import defaultdict

import numpy as np

from collectors.pronosoft_history import load_history, get_history_path
from models.odds_predictor import OddsPredictor, DEFAULT_WEIGHTS


def backtest_odds_model(history_path: str | None = None,
                        weights: dict = None,
                        grid_type: str = "LF7") -> dict:
    """Backteste le modèle OddsPredictor sur l'historique JSON.

    Args:
        history_path: chemin vers le fichier JSON d'historique (auto si None)
        weights: poids du modèle (None = poids par défaut)
        grid_type: type de grille (LF7, LF8, LF12, LF15)

    Returns:
        dict avec métriques globales et détails par grille
    """
    if history_path is None:
        history_path = get_history_path(grid_type)
    grids = load_history(history_path, grid_type=grid_type)
    if not grids:
        return {"error": "Aucune grille chargée", "n_grilles": 0}

    predictor = OddsPredictor(weights=weights)

    total_correct = 0
    total_matchs = 0
    grid_results = []
    correct_distribution = defaultdict(int)

    for grid in grids:
        matches = grid.get("matches", [])
        resultats = grid.get("resultats", "")
        if not matches or not resultats or len(resultats) != len(matches):
            continue

        predictions = predictor.predict_grid(grid)
        grid_correct = 0
        match_details = []

        for i, pred in enumerate(predictions):
            if i >= len(resultats):
                break
            actual = resultats[i]
            predicted = pred["prediction"]
            correct = predicted == actual

            if correct:
                grid_correct += 1
                total_correct += 1
            total_matchs += 1

            match_details.append({
                "position": i + 1,
                "home": pred.get("home", matches[i].get("home", "")),
                "away": pred.get("away", matches[i].get("away", "")),
                "predicted": predicted,
                "actual": actual,
                "correct": correct,
                "prob_1": pred["prob_1"],
                "prob_n": pred["prob_n"],
                "prob_2": pred["prob_2"],
                "confiance": pred["confiance"],
            })

        correct_distribution[grid_correct] += 1

        grid_results.append({
            "grid_number": grid.get("grid_number"),
            "difficulty": grid.get("difficulty"),
            "resultats_reels": resultats,
            "resultats_predits": "".join(m["predicted"] for m in match_details),
            "n_correct": grid_correct,
            "is_perfect": grid_correct == len(matches),
            "matches": match_details,
        })

    n_grilles = len(grid_results)
    n_parfaites = sum(1 for g in grid_results if g["is_perfect"])
    n_6_plus = sum(1 for g in grid_results if g["n_correct"] >= 6)
    accuracy = total_correct / total_matchs if total_matchs > 0 else 0.0
    moy_correct = (
        sum(g["n_correct"] for g in grid_results) / n_grilles
        if n_grilles > 0 else 0.0
    )

    # Determine nb_matchs from data
    nb_matchs = 7
    if grid_results:
        for g in grids:
            nm = g.get("nb_matchs") or len(g.get("matches", []))
            if nm:
                nb_matchs = nm
                break

    return {
        "n_grilles": n_grilles,
        "n_matchs": total_matchs,
        "nb_matchs": nb_matchs,
        "accuracy": accuracy,
        "moy_correct_par_grille": round(moy_correct, 2),
        f"n_parfaites_{nb_matchs}_sur_{nb_matchs}": n_parfaites,
        "n_parfaites": n_parfaites,
        "n_6_plus": n_6_plus,
        "pct_parfaites": round(n_parfaites / n_grilles * 100, 2) if n_grilles > 0 else 0,
        "pct_6_plus": round(n_6_plus / n_grilles * 100, 2) if n_grilles > 0 else 0,
        "correct_distribution": dict(sorted(correct_distribution.items())),
        "grids": grid_results,
    }


def compare_strategies(history_path: str | None = None,
                       grid_type: str = "LF7") -> dict:
    """Compare les trois stratégies sur l'historique.

    Returns:
        dict avec les résultats pour chaque stratégie
    """
    if history_path is None:
        history_path = get_history_path(grid_type)
    grids = load_history(history_path, grid_type=grid_type)
    if not grids:
        return {"error": "Aucune grille chargée"}

    results = {}

    # Différentes configurations de poids à tester
    configs = {
        "prudente": {
            "cotes": 0.70, "consensus": 0.10, "cyborg": 0.05,
            "difficulte": 0.10, "historique": 0.05,
        },
        "equilibree": DEFAULT_WEIGHTS.copy(),
        "audacieuse": {
            "cotes": 0.45, "consensus": 0.20, "cyborg": 0.15,
            "difficulte": 0.10, "historique": 0.10,
        },
    }

    for strategy_name, weights in configs.items():
        predictor = OddsPredictor(strategy=strategy_name, weights=weights)

        total_correct = 0
        total_matchs = 0
        n_7 = 0
        n_6_plus = 0

        for grid in grids:
            matches = grid.get("matches", [])
            resultats = grid.get("resultats", "")
            if not matches or not resultats or len(resultats) != len(matches):
                continue

            predictions = predictor.predict_grid(grid)
            grid_correct = 0
            for i, pred in enumerate(predictions):
                if i < len(resultats) and pred["prediction"] == resultats[i]:
                    grid_correct += 1
                    total_correct += 1
                total_matchs += 1

            nb_m = len(matches)
            if grid_correct == nb_m:
                n_7 += 1
            if grid_correct >= nb_m - 1:
                n_6_plus += 1

        # Determine nb_matchs from data
        nb_matchs = 7
        for g in grids:
            nm = g.get("nb_matchs") or len(g.get("matches", []))
            if nm:
                nb_matchs = nm
                break
        n_grilles = total_matchs // nb_matchs if total_matchs > 0 else 0
        results[strategy_name] = {
            "weights": weights,
            "accuracy": total_correct / total_matchs if total_matchs > 0 else 0.0,
            f"n_{nb_matchs}_sur_{nb_matchs}": n_7,
            "n_parfaites": n_7,
            f"n_{nb_matchs - 1}_plus": n_6_plus,
            "n_6_plus": n_6_plus,
            "n_grilles": n_grilles,
            "nb_matchs": nb_matchs,
        }

    return results


def run_full_backtest(history_path: str | None = None,
                      grid_type: str = "LF7") -> dict:
    """Rapport complet de backtesting.

    Returns:
        dict avec tous les résultats
    """
    if history_path is None:
        history_path = get_history_path(grid_type)

    print("=" * 70)
    print(f"BACKTESTING COMPLET {grid_type} — Modèle basé sur les cotes")
    print("=" * 70)

    # 1. Backtest avec poids par défaut
    print("\n--- Backtest poids par défaut ---")
    bt = backtest_odds_model(history_path, grid_type=grid_type)
    if "error" in bt:
        print(f"  Erreur: {bt['error']}")
        return bt

    nb = bt.get("nb_matchs", 7)
    print(f"  Grilles: {bt['n_grilles']}")
    print(f"  Matchs: {bt['n_matchs']}")
    print(f"  Accuracy: {bt['accuracy']:.4f}")
    print(f"  Moyenne correct/grille: {bt['moy_correct_par_grille']}/{nb}")
    print(f"  Grilles {nb}/{nb}: {bt['n_parfaites']} ({bt['pct_parfaites']}%)")
    print(f"  Grilles {nb-1}+/{nb}: {bt['n_6_plus']} ({bt['pct_6_plus']}%)")
    print(f"  Distribution:")
    for n_ok, count in bt["correct_distribution"].items():
        bar = "#" * min(count, 50)
        print(f"    {n_ok}/{nb}: {count:3d} {bar}")

    # 2. Comparaison de stratégies
    print("\n" + "-" * 70)
    print("COMPARAISON DES STRATÉGIES")
    print("-" * 70)

    strategies = compare_strategies(history_path, grid_type=grid_type)
    if "error" not in strategies:
        for name, data in strategies.items():
            snb = data.get("nb_matchs", nb)
            print(f"\n  {name.upper()}:")
            print(f"    Accuracy: {data['accuracy']:.4f}")
            print(f"    {snb}/{snb}: {data['n_parfaites']}")
            print(f"    {snb-1}+/{snb}: {data['n_6_plus']}")

    # 3. Optimisation des poids
    print("\n" + "-" * 70)
    print("OPTIMISATION DES POIDS")
    print("-" * 70)

    try:
        from models.weight_optimizer import WeightOptimizer
        optimizer = WeightOptimizer(history_path)
        opt_result = optimizer.optimize_global(metric="accuracy", n_restarts=5)
        print(f"\n  Meilleurs poids (accuracy): {opt_result['weights']}")
        print(f"  Score: {opt_result['score']:.4f}")

        # Backtest with optimized weights
        bt_opt = backtest_odds_model(history_path, weights=opt_result["weights"],
                                     grid_type=grid_type)
        print(f"  Accuracy optimisée: {bt_opt['accuracy']:.4f}")
        print(f"  {nb}/{nb} optimisés: {bt_opt['n_parfaites']}")
        print(f"  {nb-1}+/{nb} optimisés: {bt_opt['n_6_plus']}")
    except Exception as e:
        print(f"  Optimisation échouée: {e}")
        opt_result = None
        bt_opt = None

    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"\n  Poids par défaut: accuracy={bt['accuracy']:.4f}, "
          f"{nb}/{nb}={bt['n_parfaites']}, {nb-1}+/{nb}={bt['n_6_plus']}")
    if bt_opt:
        print(f"  Poids optimisés: accuracy={bt_opt['accuracy']:.4f}, "
              f"{nb}/{nb}={bt_opt['n_parfaites']}, {nb-1}+/{nb}={bt_opt['n_6_plus']}")

    return {
        "default": bt,
        "strategies": strategies,
        "optimized": opt_result,
        "optimized_backtest": bt_opt,
    }
