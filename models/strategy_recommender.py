"""Recommandation de strategie pre-match.

Analyse les performances historiques des 3 strategies (prudente/equilibree/audacieuse)
et construit des regles simples pour recommander la meilleure strategie
en fonction des metriques de la grille courante.
"""

import json
import os
from collections import defaultdict

import numpy as np
from loguru import logger

from collectors.pronosoft_history import load_history, DEFAULT_HISTORY_PATH
from models.odds_predictor import OddsPredictor, DEFAULT_WEIGHTS


STRATEGY_WEIGHTS = {
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

RULES_PATH = os.path.join("data", "models", "strategy_rules.json")


def build_strategy_performance_table(history_path: str = DEFAULT_HISTORY_PATH) -> list[dict]:
    """Backteste les 3 strategies sur chaque grille historique.

    Pour chaque grille, cree un predicteur par strategie, predit les 7 matchs,
    compare avec les resultats reels, et determine la meilleure strategie.

    Returns:
        Liste de dicts avec metriques et performances par strategie par grille.
    """
    grids = load_history(history_path)
    if not grids:
        logger.error("Aucune grille chargee pour l'analyse de strategies.")
        return []

    predictors = {
        name: OddsPredictor(strategy=name, weights=weights)
        for name, weights in STRATEGY_WEIGHTS.items()
    }

    table = []

    for grid in grids:
        matches = grid.get("matches", [])
        resultats = grid.get("resultats", "")
        if not matches or not resultats or len(resultats) != len(matches):
            continue

        row = {
            "grid_number": grid.get("grid_number"),
            "inv_spread_sum": grid.get("inv_spread_sum"),
            "std_cote_fav": grid.get("std_cote_fav"),
            "accord": grid.get("accord_cotes_joueurs"),
            "difficulty": grid.get("difficulty"),
            "moyenne_cote_fav": grid.get("moyenne_cote_fav"),
            "nb_matchs_serres": grid.get("nb_matchs_serres"),
        }

        scores = {}
        for name, predictor in predictors.items():
            predictions = predictor.predict_grid(grid)
            n_correct = sum(
                1 for i, pred in enumerate(predictions)
                if i < len(resultats) and pred["prediction"] == resultats[i]
            )
            col = f"n_correct_{name}"
            row[col] = n_correct
            scores[name] = n_correct

        best = max(scores, key=scores.get)
        best_score = scores[best]
        second_best = sorted(scores.values(), reverse=True)[1]
        row["best_strategy"] = best
        row["margin"] = best_score - second_best

        table.append(row)

    logger.info(f"Table de performance construite: {len(table)} grilles analysees.")
    return table


def build_recommendation_rules(perf_table: list[dict]) -> dict:
    """Construit des regles de recommandation a partir de la table de performance.

    Scanne des seuils sur les metriques cles et identifie les regles
    qui discriminent le mieux la strategie gagnante.

    Returns:
        dict avec 'rules' (liste de regles) et 'accuracy' globale.
    """
    if not perf_table:
        return {"rules": [], "accuracy": 0.0}

    metrics = ["inv_spread_sum", "std_cote_fav", "accord", "difficulty"]
    best_rules = []

    for metric in metrics:
        values = [r[metric] for r in perf_table if r.get(metric) is not None]
        if not values:
            continue

        percentiles = np.percentile(values, [25, 33, 50, 67, 75])
        thresholds = sorted(set(percentiles))

        for threshold in thresholds:
            # Regle: si metric < seuil -> quelle strategie est la meilleure?
            low_group = [r for r in perf_table if r.get(metric) is not None and r[metric] < threshold]
            high_group = [r for r in perf_table if r.get(metric) is not None and r[metric] >= threshold]

            for group, direction in [(low_group, "low"), (high_group, "high")]:
                if len(group) < 5:
                    continue

                strategy_wins = defaultdict(int)
                for r in group:
                    strategy_wins[r["best_strategy"]] += 1

                best_strat = max(strategy_wins, key=strategy_wins.get)
                accuracy = strategy_wins[best_strat] / len(group)

                if accuracy >= 0.40:
                    best_rules.append({
                        "metric": metric,
                        "threshold": round(float(threshold), 4),
                        "direction": direction,
                        "recommended": best_strat,
                        "accuracy": round(accuracy, 4),
                        "support": len(group),
                    })

    # Trier par accuracy decroissante et garder les meilleures regles
    best_rules.sort(key=lambda r: (r["accuracy"], r["support"]), reverse=True)
    # Garder max 2 regles par metrique pour eviter la redondance
    selected = []
    metric_count = defaultdict(int)
    for rule in best_rules:
        if metric_count[rule["metric"]] < 2:
            selected.append(rule)
            metric_count[rule["metric"]] += 1
        if len(selected) >= 8:
            break

    # Calculer l'accuracy globale des regles combinees
    correct = 0
    total = 0
    for row in perf_table:
        prediction = _apply_rules(row, selected)
        if prediction and prediction == row["best_strategy"]:
            correct += 1
        total += 1

    global_accuracy = correct / total if total > 0 else 0.0

    result = {
        "rules": selected,
        "accuracy": round(global_accuracy, 4),
        "n_grids": len(perf_table),
    }

    # Sauvegarder
    os.makedirs(os.path.dirname(RULES_PATH), exist_ok=True)
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Regles sauvegardees dans {RULES_PATH} (accuracy={global_accuracy:.1%})")

    return result


def _apply_rules(grid_metrics: dict, rules: list[dict]) -> str | None:
    """Applique les regles et retourne la strategie recommandee par vote."""
    votes = defaultdict(float)

    for rule in rules:
        val = grid_metrics.get(rule["metric"])
        if val is None:
            continue

        match = False
        if rule["direction"] == "low" and val < rule["threshold"]:
            match = True
        elif rule["direction"] == "high" and val >= rule["threshold"]:
            match = True

        if match:
            votes[rule["recommended"]] += rule["accuracy"]

    if not votes:
        return None
    return max(votes, key=votes.get)


def recommend_strategy(grid_metrics: dict, rules: list[dict] | None = None) -> dict:
    """Recommande la meilleure strategie pour une grille donnee.

    Args:
        grid_metrics: dict avec inv_spread_sum, std_cote_fav, accord, difficulty, etc.
        rules: regles pre-calculees (charge depuis fichier si None)

    Returns:
        dict {recommended, confidence, reasoning, scores}
    """
    if rules is None:
        rules = _load_rules()

    if not rules:
        return {
            "recommended": "equilibree",
            "confidence": 0.0,
            "reasoning": "Aucune regle disponible, strategie par defaut.",
            "scores": {"prudente": 0, "equilibree": 0, "audacieuse": 0},
        }

    # Calculer les votes ponderes
    votes = defaultdict(float)
    reasons = []

    for rule in rules:
        val = grid_metrics.get(rule["metric"])
        if val is None:
            continue

        match = False
        if rule["direction"] == "low" and val < rule["threshold"]:
            match = True
        elif rule["direction"] == "high" and val >= rule["threshold"]:
            match = True

        if match:
            votes[rule["recommended"]] += rule["accuracy"]
            direction_label = "<" if rule["direction"] == "low" else ">="
            reasons.append(
                f"{rule['metric']} {direction_label} {rule['threshold']:.2f} "
                f"-> {rule['recommended']} ({rule['accuracy']:.0%})"
            )

    if not votes:
        return {
            "recommended": "equilibree",
            "confidence": 0.0,
            "reasoning": "Aucune regle applicable, strategie par defaut.",
            "scores": {"prudente": 0, "equilibree": 0, "audacieuse": 0},
        }

    recommended = max(votes, key=votes.get)
    total_votes = sum(votes.values())
    confidence = votes[recommended] / total_votes if total_votes > 0 else 0.0

    # Normaliser les scores
    scores = {
        name: round(votes.get(name, 0) / total_votes, 3) if total_votes > 0 else 0
        for name in ["prudente", "equilibree", "audacieuse"]
    }

    return {
        "recommended": recommended,
        "confidence": round(confidence, 3),
        "reasoning": " | ".join(reasons[:3]),
        "scores": scores,
    }


def _load_rules() -> list[dict]:
    """Charge les regles depuis le fichier JSON."""
    if not os.path.exists(RULES_PATH):
        return []
    try:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("rules", [])
    except (json.JSONDecodeError, OSError):
        return []
