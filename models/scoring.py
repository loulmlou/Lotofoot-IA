"""Moteur de scoring pondéré — combine tous les signaux pour une prédiction finale."""

import numpy as np

from analysis.baseline import LABEL_NAMES
from config.settings import SCORING_WEIGHTS


def _normalize(scores):
    """Normalise un vecteur [s1, sn, s2] pour que la somme = 1.

    Si toutes les valeurs sont 0 ou négatives, retourne une distribution uniforme.
    """
    scores = np.array(scores, dtype=float)
    scores = np.maximum(scores, 0.0)
    total = scores.sum()
    if total <= 0:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    return scores / total


def _softmax_3way(strength_diff, draw_sensitivity=6.0):
    """Convertit une différence de force en probabilités 1/N/2.

    Quand la différence est faible, la probabilité de nul augmente.
    Quand elle est forte, le favori domine et le nul diminue.
    Calibré sur la distribution réelle du football européen (~43/27/30).

    Args:
        strength_diff: différence de force (positif = dom plus fort),
                       typiquement dans [-0.5, +0.5]
        draw_sensitivity: contrôle la largeur du pic de nul.

    Returns:
        np.array [p1, pn, p2] normalisé à 1
    """
    abs_diff = abs(strength_diff)

    # Nul : pic à ~0.36 quand diff=0, descend à ~0.24 pour grands écarts
    pn = 0.24 + 0.12 * np.exp(-draw_sensitivity * abs_diff * abs_diff)

    # Répartir le reste entre 1 et 2 via sigmoid
    remaining = 1.0 - pn
    sigmoid = 1.0 / (1.0 + np.exp(-4.0 * strength_diff))
    # Léger ajustement négatif pour compenser le biais pro-dom résiduel
    # des composants forme/classement (les cotes capturent déjà l'avantage dom)
    ratio = np.clip(sigmoid - 0.025, 0.05, 0.95)

    p1 = remaining * ratio
    p2 = remaining * (1.0 - ratio)

    return _normalize([p1, pn, p2])


def score_cotes(features):
    """Score basé sur les cotes — probabilités implicites normalisées.

    Args:
        features: dict avec prob_1, prob_n, prob_2 (ou cote_1, cote_n, cote_2)

    Returns:
        np.array [p1, pn, p2] normalisé à 1
    """
    p1 = features.get("prob_1")
    pn = features.get("prob_n")
    p2 = features.get("prob_2")

    if p1 is not None and pn is not None and p2 is not None:
        if not (np.isnan(p1) or np.isnan(pn) or np.isnan(p2)):
            return _normalize([p1, pn, p2])

    # Fallback: calculer depuis les cotes brutes
    c1 = features.get("cote_1")
    cn = features.get("cote_n")
    c2 = features.get("cote_2")

    if c1 and cn and c2 and c1 > 0 and cn > 0 and c2 > 0:
        return _normalize([1.0 / c1, 1.0 / cn, 1.0 / c2])

    return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])


def score_forme(features):
    """Score basé sur la forme récente des équipes.

    Quand les deux équipes ont une forme similaire, la probabilité de nul
    augmente.

    Args:
        features: dict avec dom_forme_pts, dom_forme_ratio_vic, dom_forme_buts_m/e,
                  ext_forme_pts, ext_forme_ratio_vic, ext_forme_buts_m/e

    Returns:
        np.array [p1, pn, p2] normalisé à 1
    """
    dom_pts = features.get("dom_forme_pts", 0) or 0
    ext_pts = features.get("ext_forme_pts", 0) or 0
    dom_vic = features.get("dom_forme_ratio_vic", 0) or 0
    ext_vic = features.get("ext_forme_ratio_vic", 0) or 0
    dom_bm = features.get("dom_forme_buts_m", 0) or 0
    dom_be = features.get("dom_forme_buts_e", 0) or 0
    ext_bm = features.get("ext_forme_buts_m", 0) or 0
    ext_be = features.get("ext_forme_buts_e", 0) or 0

    # Score composite normalisé [0, 1]
    dom_score = dom_pts / 15.0 * 0.4 + dom_vic * 0.3 + (dom_bm - dom_be + 2) / 4.0 * 0.3
    ext_score = ext_pts / 15.0 * 0.4 + ext_vic * 0.3 + (ext_bm - ext_be + 2) / 4.0 * 0.3

    diff = dom_score - ext_score
    return _softmax_3way(diff)


def score_classement(features):
    """Score basé sur le classement (positions et points).

    Quand les positions sont proches, le nul est favorisé.

    Args:
        features: dict avec dom_position, ext_position, diff_position,
                  dom_classement_pts, ext_classement_pts

    Returns:
        np.array [p1, pn, p2] normalisé à 1
    """
    dom_pos = features.get("dom_position")
    ext_pos = features.get("ext_position")
    dom_pts = features.get("dom_classement_pts", 0) or 0
    ext_pts = features.get("ext_classement_pts", 0) or 0

    if dom_pos and ext_pos and not (np.isnan(dom_pos) or np.isnan(ext_pos)):
        pos_factor = np.clip((ext_pos - dom_pos) / 10.0, -1.0, 1.0)
    else:
        pts_total = max(dom_pts + ext_pts, 1)
        pos_factor = (dom_pts - ext_pts) / pts_total

    return _softmax_3way(pos_factor * 0.5)


def score_historique(features):
    """Score basé sur l'historique des confrontations directes (H2H).

    Utilise directement les pourcentages H2H quand disponibles,
    mélangés avec des priors uniformes.

    Args:
        features: dict avec h2h_pct_dom, h2h_pct_nul, h2h_pct_ext, h2h_nb

    Returns:
        np.array [p1, pn, p2] normalisé à 1
    """
    h2h_nb = features.get("h2h_nb", 0) or 0
    h2h_dom = features.get("h2h_pct_dom", 0) or 0
    h2h_nul = features.get("h2h_pct_nul", 0) or 0
    h2h_ext = features.get("h2h_pct_ext", 0) or 0

    if h2h_nb == 0:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    confidence = min(h2h_nb / 5.0, 1.0)
    prior = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    h2h = np.array([h2h_dom, h2h_nul, h2h_ext])
    blended = h2h * confidence + prior * (1 - confidence)

    return _normalize(blended)


def score_lotofoot(features, grid_type=None):
    """Score basé sur les biais historiques des grilles Loto Foot.

    Args:
        features: dict (pas utilisé pour le moment, prêt pour évolution)
        grid_type: type de grille ('LF7', 'LF8', 'LF15', 'LF12')

    Returns:
        np.array [p1, pn, p2] normalisé à 1
    """
    distributions = {
        "LF7":  [0.43, 0.27, 0.30],
        "LF8":  [0.43, 0.27, 0.30],
        "LF15": [0.44, 0.27, 0.29],
        "LF12": [0.43, 0.28, 0.29],
    }

    if grid_type and grid_type in distributions:
        return np.array(distributions[grid_type])

    return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])


def score_modele_ia(features, model=None, preprocessor=None, feature_cols=None):
    """Score basé sur les probabilités du modèle ML (XGBoost/LightGBM).

    Args:
        features: dict de features du match
        model: modèle entraîné avec predict_proba
        preprocessor: pipeline de prétraitement
        feature_cols: colonnes de features attendues par le modèle

    Returns:
        np.array [p1, pn, p2] normalisé à 1
    """
    if model is None or preprocessor is None:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    if feature_cols is None:
        from models.train import FEATURE_COLS
        feature_cols = FEATURE_COLS

    import pandas as pd
    row = {c: features.get(c, np.nan) for c in feature_cols}
    X = pd.DataFrame([row])[feature_cols]
    X_processed = preprocessor.transform(X)

    proba = model.predict_proba(X_processed)[0]
    return _normalize(proba)


def score_contexte(features):
    """Score basé sur le contexte : avantage domicile de la ligue + jours de repos.

    Args:
        features: dict avec avantage_dom_ligue, dom_jours_repos, ext_jours_repos

    Returns:
        np.array [p1, pn, p2] normalisé à 1
    """
    adv_dom = features.get("avantage_dom_ligue")
    dom_repos = features.get("dom_jours_repos")
    ext_repos = features.get("ext_jours_repos")

    # L'avantage domicile de la ligue (0.35 à 0.55 typiquement)
    # Centrer sur la moyenne globale (0.44) pour que le signal soit symétrique
    if adv_dom is not None and not np.isnan(adv_dom):
        # Au-dessus de 0.44 = ligue avec fort avantage dom, en-dessous = faible
        strength_diff = (adv_dom - 0.44) * 2.0
    else:
        strength_diff = 0.0  # pas d'info → neutre

    # Ajustement repos
    if dom_repos is not None and ext_repos is not None:
        if not (np.isnan(dom_repos) or np.isnan(ext_repos)):
            repos_diff = np.clip((dom_repos - ext_repos) / 7.0, -0.3, 0.3)
            strength_diff += repos_diff * 0.1

    return _softmax_3way(strength_diff)


def _apply_draw_correction(final):
    """Corrige le biais anti-nul dans les probabilités combinées.

    Principe : quand p1 et p2 sont proches, les signaux ne distinguent pas
    clairement le favori, ce qui en football signifie que le nul est plus
    probable que ce que les composants individuels estiment.

    Cette correction transfère de la probabilité de 1 et 2 vers N
    proportionnellement à la proximité entre p1 et p2.
    """
    p1, pn, p2 = final[0], final[1], final[2]

    # Mesure de proximité entre p1 et p2 : 1 quand identiques, 0 quand très différents
    p12_sum = p1 + p2
    if p12_sum < 0.01:
        return final

    closeness = 1.0 - abs(p1 - p2) / p12_sum
    # closeness vaut 1 quand p1=p2 (match très équilibré), 0 quand un favori clair

    # Transférer vers le nul quand les équipes sont proches
    transfer = closeness * 0.043
    final[1] += transfer
    # Retirer proportionnellement de p1 et p2
    final[0] -= transfer * (p1 / p12_sum)
    final[2] -= transfer * (p2 / p12_sum)

    return final


def compute_final_score(features, model=None, preprocessor=None,
                        feature_cols=None, weights=None, grid_type=None):
    """Calcule le score final pondéré en combinant tous les signaux.

    Args:
        features: dict de features du match
        model: modèle ML entraîné (optionnel)
        preprocessor: pipeline sklearn (optionnel)
        feature_cols: colonnes features pour le modèle (optionnel)
        weights: dict de poids (défaut: SCORING_WEIGHTS)
        grid_type: type de grille Loto Foot (optionnel)

    Returns:
        dict {
            prob_1: float,
            prob_n: float,
            prob_2: float,
            prediction: str ('1', 'N', ou '2'),
            confiance: float (0-1),
            detail_scores: dict des scores par composant
        }
    """
    if weights is None:
        weights = SCORING_WEIGHTS

    scores = {
        "cotes": score_cotes(features),
        "forme": score_forme(features),
        "classement": score_classement(features),
        "historique": score_historique(features),
        "stats_lotofoot": score_lotofoot(features, grid_type=grid_type),
        "modele_ia": score_modele_ia(features, model, preprocessor, feature_cols),
        "contexte": score_contexte(features),
    }

    # Somme pondérée
    final = np.zeros(3)
    total_weight = 0.0

    for component, weight in weights.items():
        if component in scores:
            final += weight * scores[component]
            total_weight += weight

    if total_weight > 0:
        final = final / total_weight

    # Correction du biais anti-nul : quand p1 ≈ p2, boost le nul
    final = _apply_draw_correction(final)

    final = _normalize(final)

    pred_idx = int(np.argmax(final))
    prediction = LABEL_NAMES[pred_idx]

    sorted_probs = np.sort(final)[::-1]
    confiance = sorted_probs[0] - sorted_probs[1]

    detail = {k: {"prob_1": float(v[0]), "prob_n": float(v[1]), "prob_2": float(v[2])}
              for k, v in scores.items()}

    return {
        "prob_1": float(final[0]),
        "prob_n": float(final[1]),
        "prob_2": float(final[2]),
        "prediction": prediction,
        "confiance": float(confiance),
        "detail_scores": detail,
    }
