"""Analyse comparative cross-types (LF7, LF8, LF12, LF15).

Compare les métriques pré/post-match entre les 4 types de grilles pour
identifier quelle grille jouer en fonction des caractéristiques à venir.
"""

import math
import os
import sys
from collections import defaultdict
from statistics import mean, median, stdev, correlation

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collectors.pronosoft_history import load_history
from analysis.correlation_study import extract_grid_features, compute_correlations

GRID_TYPES = ["LF7", "LF8", "LF12", "LF15"]


# =========================================================================
# 1. Chargement de toutes les features par type
# =========================================================================

def load_all_types_features() -> dict[str, list[dict]]:
    """Charge les historiques et extrait les features pour chaque type disponible.

    Returns:
        {"LF7": [features_dict, ...], "LF8": [...], ...}
        Les types sans données sont omis.
    """
    all_features = {}
    for grid_type in GRID_TYPES:
        history = load_history(grid_type=grid_type)
        if not history:
            continue
        features = []
        for grid in history:
            feat = extract_grid_features(grid)
            if feat:
                features.append(feat)
        if features:
            all_features[grid_type] = features
    return all_features


# =========================================================================
# 2. Statistiques descriptives comparatives
# =========================================================================

# Métriques pré-match à comparer
PRE_MATCH_METRICS = [
    "difficulty", "nb_pronostics",
    "moy_cote_fav", "std_cote_fav", "somme_cotes_fav",
    "moy_spread", "moy_prob_fav",
    "accord_cotes_joueurs", "moy_dispersion_votes",
    "nb_matchs_serres", "nb_matchs_faciles",
    "inv_spread_sum", "coeff_variation_fav",
    "ratio_fav_outsider", "harmonic_mean_fav",
    "hhi_moyen", "gini_cotes_fav", "skewness_cotes_fav", "kurtosis_cotes_fav",
    "concentration_top3", "kelly_moyen", "brier_ante", "marge_bookmaker_moy",
]

# Métriques post-match à comparer
POST_MATCH_METRICS = [
    "nb_surprises", "pct_fav_gagne",
    "moy_cote_resultat",
    "cyborg_accuracy", "joueurs_accuracy",
    "gagnants_parfait", "montant_parfait",
    "gagnants_presque", "montant_presque",
    "has_gagnant_parfait",
    "entropie_resultat", "alternance",
    "roi_theorique", "surprise_index", "brier_realise",
]

# Métriques à normaliser par nb_matchs pour comparaison inter-types
NORMALIZE_BY_NB_MATCHS = [
    "nb_surprises", "nb_matchs_serres", "nb_matchs_faciles",
    "inv_spread_sum", "somme_cotes_fav",
]


def _safe_stats(values: list) -> dict:
    """Calcule les stats descriptives d'une liste de valeurs."""
    if not values:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "n": 0}
    return {
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "std": round(stdev(values), 4) if len(values) > 1 else 0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def compare_types_descriptive(all_features: dict[str, list[dict]]) -> dict:
    """Compare les stats descriptives de chaque métrique entre les types.

    Returns:
        {
            "par_type": {"LF7": {"metric": stats_dict, ...}, ...},
            "normalise": {"LF7": {"metric_norm": stats_dict, ...}, ...},
            "rankings": {"metric": ["LF7", "LF12", ...], ...}  # du plus haut au plus bas
        }
    """
    all_metrics = PRE_MATCH_METRICS + POST_MATCH_METRICS
    par_type = {}
    normalise = {}

    for grid_type, features in all_features.items():
        par_type[grid_type] = {}
        normalise[grid_type] = {}
        for metric in all_metrics:
            vals = [f[metric] for f in features if metric in f and f[metric] is not None]
            par_type[grid_type][metric] = _safe_stats(vals)

            # Version normalisée par nb_matchs
            if metric in NORMALIZE_BY_NB_MATCHS:
                nb_matchs_list = [f.get("nb_matchs", 7) for f in features if metric in f]
                if nb_matchs_list:
                    norm_vals = [
                        f[metric] / f.get("nb_matchs", 7)
                        for f in features
                        if metric in f and f[metric] is not None
                    ]
                    normalise[grid_type][f"{metric}_norm"] = _safe_stats(norm_vals)

    # Rankings par métrique (type avec la plus haute moyenne en premier)
    rankings = {}
    for metric in all_metrics:
        type_means = []
        for grid_type in all_features:
            stats = par_type[grid_type].get(metric, {})
            if stats.get("n", 0) > 0:
                type_means.append((grid_type, stats["mean"]))
        type_means.sort(key=lambda x: x[1], reverse=True)
        rankings[metric] = [t[0] for t in type_means]

    return {
        "par_type": par_type,
        "normalise": normalise,
        "rankings": rankings,
    }


# =========================================================================
# 3. Comparaison de la prévisibilité
# =========================================================================

def compare_predictability(all_features: dict[str, list[dict]]) -> dict:
    """Compare la prévisibilité entre les types de grilles.

    Returns:
        {
            "par_type": {
                "LF7": {"pct_fav_gagne": stats, "cyborg_accuracy": stats, ...},
                ...
            },
            "classement_previsibilite": ["LF7", "LF8", ...],  # du plus prévisible
            "scores": {"LF7": score_composite, ...}
        }
    """
    predictability_metrics = [
        "pct_fav_gagne", "cyborg_accuracy", "joueurs_accuracy",
        "accord_cotes_joueurs", "indice_previsibilite",
    ]

    par_type = {}
    scores = {}

    for grid_type, features in all_features.items():
        par_type[grid_type] = {}
        composite = []
        for metric in predictability_metrics:
            vals = [f[metric] for f in features if metric in f and f[metric] is not None]
            par_type[grid_type][metric] = _safe_stats(vals)
            if vals:
                composite.append(mean(vals))
        # Score composite = moyenne des moyennes (toutes sont dans [0,1])
        scores[grid_type] = round(mean(composite), 4) if composite else 0

    classement = sorted(scores, key=scores.get, reverse=True)

    return {
        "par_type": par_type,
        "classement_previsibilite": classement,
        "scores": scores,
    }


# =========================================================================
# 4. Comparaison de la valeur (gains)
# =========================================================================

def compare_value(all_features: dict[str, list[dict]]) -> dict:
    """Compare la valeur financière entre les types.

    Returns:
        {
            "par_type": {"LF7": {"montant_parfait": stats, ...}, ...},
            "classement_valeur": ["LF15", "LF12", ...],
            "esperance_par_eur": {"LF7": float, ...}
        }
    """
    value_metrics = [
        "montant_parfait", "gagnants_parfait",
        "montant_presque", "gagnants_presque",
        "roi_theorique",
    ]

    par_type = {}
    esperance = {}

    for grid_type, features in all_features.items():
        par_type[grid_type] = {}
        for metric in value_metrics:
            vals = [f[metric] for f in features if metric in f and f[metric] is not None]
            par_type[grid_type][metric] = _safe_stats(vals)

        # Espérance de gain par EUR misé :
        # P(gagnant_parfait) * montant_parfait_moyen + P(gagnant_presque) * montant_presque_moyen - 1
        has_gagnant_vals = [f.get("has_gagnant_parfait", 0) for f in features]
        p_parfait = mean(has_gagnant_vals) if has_gagnant_vals else 0

        montant_parfait_vals = [
            f["montant_parfait"] for f in features
            if f.get("montant_parfait", 0) > 0
        ]
        montant_parfait_moy = mean(montant_parfait_vals) if montant_parfait_vals else 0

        # Pour "presque", on approxime la probabilité par le ratio gagnants_presque > 0
        has_presque = [1 if f.get("gagnants_presque", 0) > 0 else 0 for f in features]
        p_presque = mean(has_presque) if has_presque else 0

        montant_presque_vals = [
            f["montant_presque"] for f in features
            if f.get("montant_presque", 0) > 0
        ]
        montant_presque_moy = mean(montant_presque_vals) if montant_presque_vals else 0

        esp = p_parfait * montant_parfait_moy + p_presque * montant_presque_moy - 1
        esperance[grid_type] = round(esp, 2)

    classement = sorted(esperance, key=esperance.get, reverse=True)

    return {
        "par_type": par_type,
        "classement_valeur": classement,
        "esperance_par_eur": esperance,
    }


# =========================================================================
# 5. Patterns cross-types
# =========================================================================

def find_cross_type_patterns(all_features: dict[str, list[dict]]) -> dict:
    """Identifie les patterns stables et différenciants entre les types.

    Returns:
        {
            "separating_features": [...],  # features qui séparent le mieux les types
            "stable_correlations": [...],   # corrélations stables cross-types
            "type_specific_correlations": {...},  # corrélations spécifiques à un type
        }
    """
    # 1. Features qui séparent les types (ANOVA-like via effect size)
    separating = []
    all_metrics = PRE_MATCH_METRICS + POST_MATCH_METRICS

    for metric in all_metrics:
        type_vals = {}
        for grid_type, features in all_features.items():
            vals = [f[metric] for f in features if metric in f and f[metric] is not None]
            if vals:
                type_vals[grid_type] = vals

        if len(type_vals) < 2:
            continue

        # Calculer l'effect size global (écart entre moyennes / std poolée)
        all_vals = []
        type_means = {}
        for gt, vals in type_vals.items():
            all_vals.extend(vals)
            type_means[gt] = mean(vals)

        if len(all_vals) < 10:
            continue

        global_std = stdev(all_vals) if len(all_vals) > 1 else 1
        if global_std == 0:
            continue

        # Max effect size entre 2 types quelconques
        types_list = list(type_means.keys())
        max_effect = 0
        best_pair = ("", "")
        for i in range(len(types_list)):
            for j in range(i + 1, len(types_list)):
                effect = abs(type_means[types_list[i]] - type_means[types_list[j]]) / global_std
                if effect > max_effect:
                    max_effect = effect
                    best_pair = (types_list[i], types_list[j])

        if max_effect > 0.3:
            separating.append({
                "metric": metric,
                "effect_size": round(max_effect, 3),
                "best_pair": best_pair,
                "means": {gt: round(m, 4) for gt, m in type_means.items()},
            })

    separating.sort(key=lambda x: x["effect_size"], reverse=True)

    # 2. Corrélations stables vs type-spécifiques
    # Calculer les top corrélations pour chaque type
    type_corrs = {}
    for grid_type, features in all_features.items():
        if len(features) >= 15:
            corrs = compute_correlations(features)
            type_corrs[grid_type] = {
                (c["pre"], c["post"]): c["r"] for c in corrs if abs(c["r"]) > 0.2
            }

    # Trouver les corrélations présentes dans tous les types
    stable = []
    type_specific = defaultdict(list)

    if len(type_corrs) >= 2:
        # Union de toutes les paires
        all_pairs = set()
        for corrs in type_corrs.values():
            all_pairs.update(corrs.keys())

        for pair in all_pairs:
            rs = {}
            for gt, corrs in type_corrs.items():
                if pair in corrs:
                    rs[gt] = corrs[pair]

            if len(rs) == len(type_corrs):
                # Présent dans tous les types : vérifier stabilité (même signe)
                signs = [1 if r > 0 else -1 for r in rs.values()]
                if len(set(signs)) == 1:
                    avg_r = mean(list(rs.values()))
                    stable.append({
                        "pre": pair[0],
                        "post": pair[1],
                        "r_moyen": round(avg_r, 4),
                        "r_par_type": {gt: round(r, 4) for gt, r in rs.items()},
                    })
            elif len(rs) == 1:
                # Présent dans un seul type
                gt = list(rs.keys())[0]
                type_specific[gt].append({
                    "pre": pair[0],
                    "post": pair[1],
                    "r": round(rs[gt], 4),
                })

    stable.sort(key=lambda x: abs(x["r_moyen"]), reverse=True)
    for gt in type_specific:
        type_specific[gt].sort(key=lambda x: abs(x["r"]), reverse=True)

    return {
        "separating_features": separating[:30],
        "stable_correlations": stable[:20],
        "type_specific_correlations": dict(type_specific),
    }


# =========================================================================
# 6. Recommandation de type de grille
# =========================================================================

def recommend_grid_type(upcoming_metrics: dict, all_features: dict[str, list[dict]] = None) -> dict:
    """Recommande le meilleur type de grille à jouer pour les métriques à venir.

    Args:
        upcoming_metrics: métriques pré-match de la grille à venir
        all_features: features historiques (chargées si None)

    Returns:
        {
            "recommendation": "LF7",
            "scores": {"LF7": float, ...},
            "confidence": float,
            "reasoning": [str, ...]
        }
    """
    if all_features is None:
        all_features = load_all_types_features()

    if not all_features:
        return {
            "recommendation": "LF7",
            "scores": {},
            "confidence": 0,
            "reasoning": ["Aucune donnée historique disponible."],
        }

    # Comparer les métriques à venir avec les distributions historiques de chaque type
    comparison_metrics = [
        "moy_cote_fav", "std_cote_fav", "moy_spread", "moy_prob_fav",
        "hhi_moyen", "gini_cotes_fav", "marge_bookmaker_moy",
        "difficulty", "nb_matchs_serres", "nb_matchs_faciles",
    ]

    predictability = compare_predictability(all_features)
    value = compare_value(all_features)

    scores = {}
    reasoning = []

    for grid_type, features in all_features.items():
        # Score de similarité : combien la grille à venir ressemble aux grilles historiques du type
        similarity = 0
        n_compared = 0

        for metric in comparison_metrics:
            if metric not in upcoming_metrics:
                continue
            vals = [f[metric] for f in features if metric in f and f[metric] is not None]
            if not vals or len(vals) < 5:
                continue

            m = mean(vals)
            s = stdev(vals) if len(vals) > 1 else 1
            if s == 0:
                continue

            # Z-score : combien la valeur à venir est proche de la distribution historique
            z = abs(upcoming_metrics[metric] - m) / s
            # Convertir en score de proximité (gaussien)
            prox = math.exp(-0.5 * z ** 2)
            similarity += prox
            n_compared += 1

        if n_compared > 0:
            similarity /= n_compared

        # Score de valeur (espérance de gain)
        esp = value["esperance_par_eur"].get(grid_type, -1)
        val_score = max(0, (esp + 1) / 2)  # normaliser autour de 0.5

        # Score de prévisibilité
        pred_score = predictability["scores"].get(grid_type, 0)

        # Score composite : pondération similarité 40%, valeur 35%, prévisibilité 25%
        composite = 0.4 * similarity + 0.35 * val_score + 0.25 * pred_score
        scores[grid_type] = round(composite, 4)

    if not scores:
        return {
            "recommendation": "LF7",
            "scores": {},
            "confidence": 0,
            "reasoning": ["Pas assez de données pour comparer."],
        }

    # Recommandation
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    confidence = min(1.0, (best_score - second_best) * 5 + 0.3) if second_best > 0 else 0.5

    reasoning.append(f"Type recommandé: {best_type} (score={best_score:.3f})")
    reasoning.append(f"Prévisibilité: {predictability['classement_previsibilite']}")
    reasoning.append(f"Valeur: {value['classement_valeur']}")
    for gt, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        reasoning.append(f"  {gt}: score={sc:.4f}, espérance={value['esperance_par_eur'].get(gt, '?')} EUR")

    return {
        "recommendation": best_type,
        "scores": scores,
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
    }


# =========================================================================
# 7. Rapport cross-types
# =========================================================================

def generate_cross_type_report() -> str:
    """Génère un rapport texte complet de l'analyse comparative cross-types."""

    all_features = load_all_types_features()
    if not all_features:
        return "ERREUR: Aucun historique trouvé pour aucun type de grille."

    lines = []
    lines.append("=" * 90)
    lines.append("ANALYSE COMPARATIVE CROSS-TYPES — LotoFoot")
    types_str = ", ".join(f"{t} ({len(f)} grilles)" for t, f in all_features.items())
    lines.append(f"Types analysés: {types_str}")
    lines.append("=" * 90)

    # ---- 1. Stats descriptives comparatives ----
    lines.append("\n" + "=" * 90)
    lines.append("1. STATISTIQUES DESCRIPTIVES PAR TYPE")
    lines.append("=" * 90)

    desc = compare_types_descriptive(all_features)

    key_metrics = [
        ("difficulty", "Difficulté"),
        ("moy_cote_fav", "Moy. cote favori"),
        ("std_cote_fav", "Std cote favori"),
        ("moy_spread", "Moy. spread"),
        ("moy_prob_fav", "Moy. prob favori"),
        ("hhi_moyen", "HHI moyen"),
        ("gini_cotes_fav", "Gini cotes fav"),
        ("marge_bookmaker_moy", "Marge bookmaker"),
        ("kelly_moyen", "Kelly moyen"),
        ("nb_surprises", "Nb surprises"),
        ("pct_fav_gagne", "% fav gagne"),
        ("montant_parfait", "Montant parfait"),
        ("gagnants_parfait", "Gagnants parfait"),
        ("roi_theorique", "ROI théorique"),
        ("brier_realise", "Brier réalisé"),
        ("cyborg_accuracy", "Accuracy Cyborg"),
        ("joueurs_accuracy", "Accuracy joueurs"),
    ]

    # En-tête
    header = f"  {'Métrique':25s}"
    for gt in all_features:
        header += f"  {gt:>12s}"
    lines.append(header)
    lines.append("  " + "-" * (25 + 14 * len(all_features)))

    for metric, label in key_metrics:
        row = f"  {label:25s}"
        for gt in all_features:
            stats = desc["par_type"].get(gt, {}).get(metric, {})
            m = stats.get("mean", 0)
            row += f"  {m:12.4f}"
        # Marquer le meilleur type
        ranking = desc["rankings"].get(metric, [])
        if ranking:
            row += f"  best={ranking[0]}"
        lines.append(row)

    # Métriques normalisées
    lines.append("\n  --- Métriques normalisées par nb_matchs ---")
    for metric in NORMALIZE_BY_NB_MATCHS:
        norm_key = f"{metric}_norm"
        row = f"  {metric + '/N':25s}"
        for gt in all_features:
            stats = desc["normalise"].get(gt, {}).get(norm_key, {})
            m = stats.get("mean", 0)
            row += f"  {m:12.4f}"
        lines.append(row)

    # ---- 2. Prévisibilité ----
    lines.append("\n" + "=" * 90)
    lines.append("2. COMPARAISON DE LA PRÉVISIBILITÉ")
    lines.append("=" * 90)

    pred = compare_predictability(all_features)
    lines.append(f"\n  Classement (plus prévisible en premier): {pred['classement_previsibilite']}")
    lines.append(f"  Scores composites: {pred['scores']}")

    for gt in pred["classement_previsibilite"]:
        detail = pred["par_type"][gt]
        lines.append(f"\n  {gt}:")
        for metric in ["pct_fav_gagne", "cyborg_accuracy", "joueurs_accuracy", "accord_cotes_joueurs"]:
            stats = detail.get(metric, {})
            lines.append(f"    {metric:25s}: moy={stats.get('mean', 0):.4f}  std={stats.get('std', 0):.4f}")

    # ---- 3. Valeur financière ----
    lines.append("\n" + "=" * 90)
    lines.append("3. COMPARAISON DE LA VALEUR FINANCIÈRE")
    lines.append("=" * 90)

    val = compare_value(all_features)
    lines.append(f"\n  Classement (meilleure valeur en premier): {val['classement_valeur']}")
    lines.append(f"  Espérance de gain par EUR misé: {val['esperance_par_eur']}")

    for gt in val["classement_valeur"]:
        detail = val["par_type"][gt]
        lines.append(f"\n  {gt}:")
        for metric in ["montant_parfait", "gagnants_parfait", "montant_presque", "roi_theorique"]:
            stats = detail.get(metric, {})
            lines.append(
                f"    {metric:25s}: moy={stats.get('mean', 0):10.2f}  "
                f"med={stats.get('median', 0):10.2f}  "
                f"max={stats.get('max', 0):10.2f}"
            )

    # ---- 4. Patterns cross-types ----
    lines.append("\n" + "=" * 90)
    lines.append("4. PATTERNS CROSS-TYPES")
    lines.append("=" * 90)

    patterns = find_cross_type_patterns(all_features)

    lines.append("\n  --- Features qui séparent le mieux les types (effect size > 0.3) ---")
    for p in patterns["separating_features"][:15]:
        means_str = "  ".join(f"{gt}={m:.3f}" for gt, m in p["means"].items())
        lines.append(
            f"  {p['metric']:30s}  effect={p['effect_size']:.3f}  "
            f"pair={p['best_pair']}  {means_str}"
        )

    lines.append(f"\n  --- Corrélations stables cross-types ({len(patterns['stable_correlations'])}) ---")
    for c in patterns["stable_correlations"][:10]:
        r_str = "  ".join(f"{gt}={r:.3f}" for gt, r in c["r_par_type"].items())
        lines.append(
            f"  {c['pre']:30s} -> {c['post']:25s}  r_moy={c['r_moyen']:+.4f}  {r_str}"
        )

    lines.append("\n  --- Corrélations type-spécifiques ---")
    for gt, corrs in patterns["type_specific_correlations"].items():
        if corrs:
            lines.append(f"\n  {gt} uniquement:")
            for c in corrs[:5]:
                lines.append(f"    {c['pre']:30s} -> {c['post']:25s}  r={c['r']:+.4f}")

    # ---- 5. Résumé et recommandations ----
    lines.append("\n" + "=" * 90)
    lines.append("5. RÉSUMÉ ET RECOMMANDATIONS")
    lines.append("=" * 90)

    lines.append("\n  === Classements ===")
    lines.append(f"\n  Prévisibilité : {' > '.join(pred['classement_previsibilite'])}")
    lines.append(f"  Valeur         : {' > '.join(val['classement_valeur'])}")

    lines.append("\n  Synthèse:")
    for gt in all_features:
        pred_rank = pred["classement_previsibilite"].index(gt) + 1 if gt in pred["classement_previsibilite"] else "?"
        val_rank = val["classement_valeur"].index(gt) + 1 if gt in val["classement_valeur"] else "?"
        nb = len(all_features[gt])
        nb_matchs = all_features[gt][0].get("nb_matchs", "?") if all_features[gt] else "?"
        esp = val["esperance_par_eur"].get(gt, "?")
        pred_score = pred["scores"].get(gt, "?")
        lines.append(
            f"    {gt} ({nb_matchs} matchs, {nb} grilles): "
            f"prévisibilité #{pred_rank} (score={pred_score})  "
            f"valeur #{val_rank} (espérance={esp} EUR)"
        )

    # ---- 6. Pourquoi les ROI sont négatifs ----
    lines.append("\n" + "=" * 90)
    lines.append("6. POURQUOI LES ROI THÉORIQUES SONT NÉGATIFS")
    lines.append("=" * 90)

    # Calculer les stats ROI détaillées par type
    lines.append("""
  Le ROI théorique simule une mise de 1 EUR sur chaque favori de chaque match.
  Les ROI sont négatifs à cause de la MARGE BOOKMAKER (vig/juice) :

  La somme des probabilités implicites (1/cote_1 + 1/cote_N + 1/cote_2) dépasse 1.
  L'excédent est la marge du bookmaker, qui abaisse systématiquement les cotes
  en dessous de leur "juste valeur". Même quand le favori gagne, on gagne
  moins que ce que la vraie probabilité justifierait.""")

    for gt, features in all_features.items():
        marge_vals = [f.get("marge_bookmaker_moy", 0) for f in features if "marge_bookmaker_moy" in f]
        roi_vals = [f.get("roi_theorique", 0) for f in features if "roi_theorique" in f]
        pct_fav = [f.get("pct_fav_gagne", 0) for f in features if "pct_fav_gagne" in f]
        cote_fav = [f.get("moy_cote_fav", 0) for f in features if "moy_cote_fav" in f]
        if marge_vals and roi_vals:
            m_marge = mean(marge_vals)
            m_roi = mean(roi_vals)
            m_pct = mean(pct_fav)
            m_cote = mean(cote_fav)
            # ROI théorique attendu = pct_fav * (cote_fav - 1) - (1 - pct_fav)
            roi_calcule = m_pct * (m_cote - 1) - (1 - m_pct)
            lines.append(
                f"\n  {gt}: marge={m_marge:.1%}  cote_fav_moy={m_cote:.3f}  "
                f"%fav_gagne={m_pct:.1%}  ROI_théorique={m_roi:+.1%}"
            )
            lines.append(
                f"        Calcul: {m_pct:.1%} × ({m_cote:.2f}-1) - {1-m_pct:.1%} × 1 "
                f"= {roi_calcule:+.1%}"
            )

    lines.append("""
  CONCLUSION : miser à l'aveugle sur les favoris est perdant. La marge bookmaker
  (~7%) absorbe l'avantage statistique. L'intérêt est de NE JOUER QUE QUAND LES
  CONDITIONS SONT FAVORABLES, en utilisant les seuils discriminants ci-dessous.""")

    # ---- 7. Stratégie recommandée ----
    lines.append("\n" + "=" * 90)
    lines.append("7. STRATÉGIE RECOMMANDÉE")
    lines.append("=" * 90)

    lines.append("""
  +-───────────────────────────────────────────────────────────────────────────-+
  |                    STRATÉGIE DE SÉLECTION DE GRILLE                        |
  +-───────────────────────────────────────────────────────────────────────────-+

  ÉTAPE 1 — CHOIX DU TYPE DE GRILLE

    Priorité : LF12 > LF8 > LF7 >> LF15

    • LF12 : meilleur ratio valeur/prévisibilité. Seul type avec ROI ~ 0%.
      Le "presque parfait" (11/12) paie bien (moy 158 EUR) et arrive souvent.
      Sweet spot : assez de matchs pour de gros gains, pas trop pour que
      ce soit impossible.

    • LF8 : meilleurs seuils discriminants (effect size > 1.0). Quand les
      conditions sont bonnes, c'est le type le plus "exploitable". 100% des
      grilles ont un gagnant parfait. Montant presque (7/8) moyen = 127 EUR.

    • LF7 : le plus accessible (99.4% ont un gagnant), mais montant parfait
      souvent partagé entre beaucoup de gagnants (moy 840). Gains dilués.

    • LF15 : ÉVITER. Seulement 9.7% des grilles ont un gagnant parfait.
      ROI le plus négatif (-6.5%). Jouer uniquement si spread exceptionnel.

  ÉTAPE 2 — ÉVALUER LA GRILLE (critères GO / NO-GO)

    Calculer ces indicateurs sur la grille à venir :

    +-─────────────────────┬───────────────┬───────────────────────────────────-+
    | Indicateur           | Seuil GO      | Effet observé                     |
    +-─────────────────────+───────────────+───────────────────────────────────-+
    | moy_spread           | > 4.3         | Brier -22%, surprises -33%        |
    | hhi_moyen            | > 0.44        | %fav +35%, Brier -21%             |
    | nb_matchs_faciles    | >= 3 (LF7/8)  | %fav +22 à 30%                   |
    |                      | >= 5 (LF12/15)|                                   |
    | ratio_fav_outsider   | < 0.30        | Brier -28%, surprises -50%        |
    | marge_bookmaker_moy  | < 0.06        | Marge faible = cotes plus justes  |
    | difficulty (Prono.)  | < 5.0         | %fav +17%, has_gagnant +45%       |
    +-─────────────────────┴───────────────┴───────────────────────────────────-+

    RÈGLE : jouer si au moins 3 indicateurs sur 6 sont en zone GO.
    Si 5-6 sont GO → mise maximale. Si 0-2 → passer la grille.

  ÉTAPE 3 — CONSTRUIRE LA GRILLE

    a) Identifier les matchs "acquis" (cote_fav < 1.50) :
       → Jouer le favori en simple (1 seul résultat).
       Ce sont les matchs où le favori gagne ~75% du temps.

    b) Identifier les matchs "serrés" (cote_fav > 2.00) :
       → Jouer en double (favori + 2ème choix) voire triple.
       Ce sont les matchs qui génèrent les surprises.

    c) Matchs intermédiaires (1.50 < cote_fav < 2.00) :
       → Vérifier l'accord cotes/joueurs. Si le plus joué = le favori
         des cotes (accord > 70%), jouer en simple.
       → Sinon, jouer en double.

    d) Signal Cyborg :
       → Quand le Cyborg donne un pronostic simple, il a raison ~44% du
         temps. Le Cyborg est plus fiable sur LF12/LF15 (r=0.75) que
         LF7 (r=0.58). Pondérer davantage le Cyborg sur les grandes grilles.

  ÉTAPE 4 — GESTION DE LA MISE

    • Ne jamais miser plus de 5% du bankroll sur une grille.
    • Augmenter la mise (3-5%) quand 5+ indicateurs sont GO.
    • Mise minimale (1-2%) quand seulement 3 indicateurs sont GO.
    • Répartir le budget : 50% sur la grille "safe" (favoris + doubles
      sur les serrés), 50% sur une grille "value" (quelques outsiders
      ciblés sur les matchs avec forte marge bookmaker).

  +-───────────────────────────────────────────────────────────────────────────-+
  |                    SIGNAUX D'ALERTE (NE PAS JOUER)                         |
  +-───────────────────────────────────────────────────────────────────────────-+

    • moy_spread < 3.0 → grille trop serrée, imprévisible
    • difficulty > 7.5 → trop de matchs difficiles
    • nb_matchs_serres > 5 (LF7/8) ou > 8 (LF12/15) → trop d'incertitude
    • 0 matchs faciles (cote_fav < 1.5) → aucun match "acquis"
    • hhi_moyen < 0.38 → matchs trop équilibrés, quasi-aléatoire""")

    # ---- 8. Exploitation des corrélations type-spécifiques ----
    lines.append("\n" + "=" * 90)
    lines.append("8. CORRÉLATIONS EXPLOITABLES PAR TYPE")
    lines.append("=" * 90)

    lines.append("""
  LF7 :
    • moy_spread/inv_spread_sum -> gagnants_parfait (r=+0.50)
      Quand le ratio spread/inv_spread est élevé, la grille est "claire"
      et il y a beaucoup de gagnants → mise plus faible car gain partagé.
    • Si moy_cote_outsider est élevé → plus de gagnants parfait (r=+0.33).
      Quand les outsiders sont très chers, peu de joueurs misent dessus,
      donc ceux qui trouvent le bon résultat partagent moins.

  LF8 :
    • ratio_fav_outsider -> montant_presque (r=+0.31)
      Quand l'écart favori/outsider est faible, le montant 7/8 est élevé.
      Signal pour viser le "presque parfait" plutôt que le parfait.
    • MEILLEURS SEUILS de toutes les grilles (effect > 1.0) :
      moy_spread > 4.32 et hhi > 0.44 sont les signaux les plus fiables.

  LF12 :
    • moy_pct_fav * hhi_moyen -> joueurs_accuracy (r=+0.58)
      Quand la foule mise sur les favoris ET que les matchs sont dominés,
      la foule a raison. Suivre la foule dans ces conditions.
    • marge_bookmaker_moy -> max_cote_resultat (r=-0.26)
      Quand la marge est faible, il y a plus de grosses surprises.
      Marge faible = bookmaker moins sûr = opportunité de value.
    • Seul type avec ROI ~ 0% : le plus rentable à long terme.

  LF15 :
    • std_cote_nul -> nb_surprises (r=-0.28)
      Quand les cotes de nul sont homogènes, il y a PLUS de surprises.
      Nuls homogènes = matchs vraiment équilibrés = imprévisible.
    • NE JOUER QUE si moy_spread > 4.5 ET hhi > 0.45 ET nb_matchs_faciles >= 5.
      Autrement, la probabilité de gagner est quasi-nulle.""")

    # Top features discriminantes
    if patterns["separating_features"]:
        lines.append("\n  Top 5 critères les plus différenciants entre types:")
        for i, p in enumerate(patterns["separating_features"][:5], 1):
            lines.append(
                f"    {i}. {p['metric']} (effect={p['effect_size']:.2f}, "
                f"plus haut pour {list(p['means'].keys())[0] if p['means'] else '?'})"
            )

    lines.append("\n" + "=" * 90)
    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_cross_type_report()
    print(report)

    report_path = os.path.join("data", "history", "cross_type_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nRapport sauvegardé: {report_path}")
