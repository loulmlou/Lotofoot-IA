"""Étude de corrélations avant/après match sur l'historique multi-types.

Cherche des liens entre indicateurs pré-match (cotes, répartition, difficulté,
métriques dérivées) et indicateurs post-match (résultat, surprises, gains).

Supporte LF7, LF8, LF12, LF15 avec des clés de features génériques.
Génère un rapport complet avec indicateurs classiques et farfelus.
"""

import json
import math
import os
import sys
from collections import Counter
from itertools import combinations
from statistics import mean, median, stdev, correlation

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collectors.pronosoft_history import load_history


# =========================================================================
# 1. Extraction des features par grille
# =========================================================================

def extract_grid_features(grid: dict) -> dict | None:
    """Extrait tous les indicateurs pré et post match d'une grille.

    Fonctionne pour tout type de grille (LF7, LF8, LF12, LF15).
    Les clés de rapports sont génériques : gagnants_parfait, montant_parfait, etc.
    """
    matches = grid.get("matches", [])
    resultats = grid.get("resultats", "")
    nb_matchs = len(matches)

    if not matches or not resultats or len(resultats) < nb_matchs:
        return None

    f = {"grid_number": grid.get("grid_number")}
    f["nb_matchs"] = nb_matchs

    # Détecter le type de grille
    grid_type = grid.get("grid_type", "")
    if not grid_type:
        type_map = {7: "LF7", 8: "LF8", 12: "LF12", 15: "LF15"}
        grid_type = type_map.get(nb_matchs, f"LF{nb_matchs}")
    f["grid_type"] = grid_type

    # --- PRÉ-MATCH (indicateurs disponibles avant le tirage) ---

    # Cotes
    cotes_1 = [m.get("cote_1", 0) for m in matches]
    cotes_n = [m.get("cote_n", 0) for m in matches]
    cotes_2 = [m.get("cote_2", 0) for m in matches]
    cotes_fav = []
    cotes_outsider = []
    spreads = []  # écart favori - outsider
    cotes_nul = []

    for m in matches:
        c = [m.get("cote_1", 0), m.get("cote_n", 0), m.get("cote_2", 0)]
        valid = [x for x in c if x > 0]
        if valid:
            cotes_fav.append(min(valid))
            cotes_outsider.append(max(valid))
            spreads.append(max(valid) - min(valid))
            cotes_nul.append(m.get("cote_n", 0))

    f["difficulty"] = grid.get("difficulty", 0) or 0
    f["nb_pronostics"] = grid.get("nb_pronostics", 0) or 0

    # Stats cotes favori
    if cotes_fav:
        f["moy_cote_fav"] = mean(cotes_fav)
        f["med_cote_fav"] = median(cotes_fav)
        f["min_cote_fav"] = min(cotes_fav)
        f["max_cote_fav"] = max(cotes_fav)
        f["std_cote_fav"] = stdev(cotes_fav) if len(cotes_fav) > 1 else 0
        f["somme_cotes_fav"] = sum(cotes_fav)
        f["prod_cotes_fav"] = math.prod(cotes_fav)
    else:
        return None

    # Stats cotes outsider
    f["moy_cote_outsider"] = mean(cotes_outsider)
    f["max_cote_outsider"] = max(cotes_outsider)
    f["somme_cotes_outsider"] = sum(cotes_outsider)

    # Spread (écart favori-outsider)
    f["moy_spread"] = mean(spreads)
    f["max_spread"] = max(spreads)
    f["min_spread"] = min(spreads)
    f["std_spread"] = stdev(spreads) if len(spreads) > 1 else 0

    # Cotes nul
    f["moy_cote_nul"] = mean(cotes_nul) if cotes_nul else 0
    f["std_cote_nul"] = stdev(cotes_nul) if len(cotes_nul) > 1 else 0

    # Ratios et combinaisons farfelues
    f["ratio_fav_outsider"] = f["moy_cote_fav"] / f["moy_cote_outsider"] if f["moy_cote_outsider"] > 0 else 0
    f["ratio_nul_fav"] = f["moy_cote_nul"] / f["moy_cote_fav"] if f["moy_cote_fav"] > 0 else 0
    f["diff_x_std"] = f["difficulty"] * f["std_cote_fav"]  # difficulté * dispersion
    f["prod_min_max_fav"] = f["min_cote_fav"] * f["max_cote_fav"]  # produit extrêmes
    f["range_cote_fav"] = f["max_cote_fav"] - f["min_cote_fav"]
    f["coeff_variation_fav"] = f["std_cote_fav"] / f["moy_cote_fav"] if f["moy_cote_fav"] > 0 else 0

    # Probabilités implicites
    probs_fav = [1 / c for c in cotes_fav]
    f["moy_prob_fav"] = mean(probs_fav)
    f["min_prob_fav"] = min(probs_fav)
    f["max_prob_fav"] = max(probs_fav)
    f["somme_prob_fav"] = sum(probs_fav)
    f["entropie_cotes"] = -sum(p * math.log2(p) if p > 0 else 0 for p in probs_fav) / len(probs_fav)

    # --- Nouvelles métriques pré-match ---

    # HHI (Herfindahl-Hirschman Index) par match : sum(p_i^2) sur les probas implicites
    hhi_values = []
    marge_bookmaker_values = []
    for m in matches:
        c = [m.get("cote_1", 0), m.get("cote_n", 0), m.get("cote_2", 0)]
        valid = [x for x in c if x > 0]
        if valid:
            probs_impl = [1 / x for x in valid]
            total_prob = sum(probs_impl)
            # Normaliser pour obtenir de vraies probabilités
            probs_norm = [p / total_prob for p in probs_impl]
            hhi_values.append(sum(p ** 2 for p in probs_norm))
            # Marge bookmaker : sum(1/cote_i) - 1
            marge_bookmaker_values.append(total_prob - 1)

    f["hhi_moyen"] = mean(hhi_values) if hhi_values else 0
    f["hhi_max"] = max(hhi_values) if hhi_values else 0
    f["hhi_min"] = min(hhi_values) if hhi_values else 0

    # Marge bookmaker moyenne
    f["marge_bookmaker_moy"] = mean(marge_bookmaker_values) if marge_bookmaker_values else 0

    # Gini des cotes favoris
    if len(cotes_fav) > 1:
        sorted_fav = sorted(cotes_fav)
        n = len(sorted_fav)
        gini_num = sum((2 * (i + 1) - n - 1) * sorted_fav[i] for i in range(n))
        gini_den = n * sum(sorted_fav)
        f["gini_cotes_fav"] = gini_num / gini_den if gini_den > 0 else 0
    else:
        f["gini_cotes_fav"] = 0

    # Skewness des cotes favoris
    if len(cotes_fav) > 2:
        m_fav = mean(cotes_fav)
        s_fav = stdev(cotes_fav)
        if s_fav > 0:
            f["skewness_cotes_fav"] = sum((x - m_fav) ** 3 for x in cotes_fav) / (len(cotes_fav) * s_fav ** 3)
        else:
            f["skewness_cotes_fav"] = 0
    else:
        f["skewness_cotes_fav"] = 0

    # Kurtosis des cotes favoris
    if len(cotes_fav) > 3:
        m_fav = mean(cotes_fav)
        s_fav = stdev(cotes_fav)
        if s_fav > 0:
            f["kurtosis_cotes_fav"] = sum((x - m_fav) ** 4 for x in cotes_fav) / (len(cotes_fav) * s_fav ** 4) - 3
        else:
            f["kurtosis_cotes_fav"] = 0
    else:
        f["kurtosis_cotes_fav"] = 0

    # Concentration top-3 : somme des 3 plus grosses probas fav / somme totale
    if len(probs_fav) >= 3:
        sorted_probs = sorted(probs_fav, reverse=True)
        f["concentration_top3"] = sum(sorted_probs[:3]) / sum(probs_fav)
    else:
        f["concentration_top3"] = 1.0

    # Kelly fractionnel moyen : (p*cote - 1) / (cote - 1) sur les favoris
    kelly_values = []
    for c_fav in cotes_fav:
        p_impl = 1 / c_fav
        if c_fav > 1:
            kelly = (p_impl * c_fav - 1) / (c_fav - 1)
            kelly_values.append(kelly)
    f["kelly_moyen"] = mean(kelly_values) if kelly_values else 0

    # Brier score ex-ante : mean(sum((p_implied - indicator_fav)^2))
    # Pour chaque match, indicateur = 1 pour le favori, 0 pour les autres
    brier_ante_values = []
    for m in matches:
        c = [m.get("cote_1", 0), m.get("cote_n", 0), m.get("cote_2", 0)]
        valid = [x for x in c if x > 0]
        if valid:
            probs_impl = [1 / x if x > 0 else 0 for x in c]
            total_prob = sum(probs_impl)
            if total_prob > 0:
                probs_norm = [p / total_prob for p in probs_impl]
                # Indicateur : 1 pour le favori (plus faible cote)
                fav_idx = c.index(min(valid))
                indicators = [1 if i == fav_idx else 0 for i in range(3)]
                brier = sum((probs_norm[i] - indicators[i]) ** 2 for i in range(3))
                brier_ante_values.append(brier)
    f["brier_ante"] = mean(brier_ante_values) if brier_ante_values else 0

    # Répartition joueurs
    pcts_1 = [m.get("pct_1", 0) for m in matches]
    pcts_n = [m.get("pct_n", 0) for m in matches]
    pcts_2 = [m.get("pct_2", 0) for m in matches]

    pcts_fav = []  # % joueurs sur le favori
    pcts_outsider = []  # % joueurs sur l'outsider
    accord_cote_joueurs = []  # le favori des cotes = le plus joué ?
    for m in matches:
        c = [m.get("cote_1", 0), m.get("cote_n", 0), m.get("cote_2", 0)]
        p = [m.get("pct_1", 0), m.get("pct_n", 0), m.get("pct_2", 0)]
        valid_c = [x for x in c if x > 0]
        if valid_c and any(x > 0 for x in p):
            fav_idx = c.index(min(valid_c))
            out_idx = c.index(max(valid_c))
            pcts_fav.append(p[fav_idx])
            pcts_outsider.append(p[out_idx])
            # Le plus joué est-il le favori des cotes ?
            max_pct_idx = p.index(max(p))
            accord_cote_joueurs.append(1 if max_pct_idx == fav_idx else 0)

    if pcts_fav:
        f["moy_pct_fav"] = mean(pcts_fav)
        f["min_pct_fav"] = min(pcts_fav)
        f["max_pct_fav"] = max(pcts_fav)
        f["moy_pct_outsider"] = mean(pcts_outsider)
        f["accord_cotes_joueurs"] = mean(accord_cote_joueurs)
        f["ecart_pct_fav_outsider"] = mean(pcts_fav) - mean(pcts_outsider)
    else:
        f["moy_pct_fav"] = 0
        f["min_pct_fav"] = 0
        f["max_pct_fav"] = 0
        f["moy_pct_outsider"] = 0
        f["accord_cotes_joueurs"] = 0
        f["ecart_pct_fav_outsider"] = 0

    # Dispersion des votes joueurs (consensus vs division)
    dispersions = []
    for m in matches:
        p = [m.get("pct_1", 0), m.get("pct_n", 0), m.get("pct_2", 0)]
        if sum(p) > 0:
            total = sum(p)
            probs = [x / total for x in p]
            entropy = -sum(pr * math.log2(pr) if pr > 0 else 0 for pr in probs)
            dispersions.append(entropy / math.log2(3))
    f["moy_dispersion_votes"] = mean(dispersions) if dispersions else 0
    f["max_dispersion_votes"] = max(dispersions) if dispersions else 0
    f["min_dispersion_votes"] = min(dispersions) if dispersions else 0

    # Signal Cyborg
    cyborg_pronos = [m.get("prono_cyborg", "") for m in matches]
    nb_cyborg_simple = sum(1 for p in cyborg_pronos if p in ("1", "N", "2"))
    nb_cyborg_double = sum(1 for p in cyborg_pronos if len(p) == 2 and p != "-")
    nb_cyborg_triple = sum(1 for p in cyborg_pronos if p in ("1N2", "-", ""))
    f["nb_cyborg_simple"] = nb_cyborg_simple
    f["nb_cyborg_double"] = nb_cyborg_double
    f["nb_cyborg_triple"] = nb_cyborg_triple
    f["ratio_cyborg_simple"] = nb_cyborg_simple / nb_matchs if nb_matchs else 0

    # Combien de matchs "serrés" (cote_fav > 2.0)
    f["nb_matchs_serres"] = sum(1 for c in cotes_fav if c > 2.0)
    f["nb_matchs_faciles"] = sum(1 for c in cotes_fav if c < 1.5)
    f["nb_matchs_tres_serres"] = sum(1 for c in cotes_fav if c > 2.5)

    # Indicateur farfelu: somme des inverses des spreads
    f["inv_spread_sum"] = sum(1 / s if s > 0 else 10 for s in spreads)
    # Harmonic mean des cotes fav
    f["harmonic_mean_fav"] = len(cotes_fav) / sum(1 / c for c in cotes_fav) if cotes_fav else 0

    # Position du match le plus serré et le plus facile
    if cotes_fav:
        f["pos_match_plus_serre"] = cotes_fav.index(max(cotes_fav)) + 1
        f["pos_match_plus_facile"] = cotes_fav.index(min(cotes_fav)) + 1

    # Symétrie de la grille
    mid = len(cotes_fav) // 2
    f["asymetrie_cotes"] = abs(mean(cotes_fav[:mid]) - mean(cotes_fav[mid:])) if mid > 0 else 0

    # Gradient de difficulté
    if len(cotes_fav) > 2:
        diffs = [cotes_fav[i + 1] - cotes_fav[i] for i in range(len(cotes_fav) - 1)]
        f["gradient_difficulte"] = mean(diffs)
    else:
        f["gradient_difficulte"] = 0

    # --- POST-MATCH (indicateurs du résultat) ---

    # Résultats
    f["nb_1"] = resultats.count("1")
    f["nb_N"] = resultats.count("N")
    f["nb_2"] = resultats.count("2")
    f["profil"] = grid.get("profil", "")

    # Surprises
    nb_surprises = 0
    nb_fav_gagne = 0
    cotes_resultats = []
    cotes_fav_match = []
    roi_gains = []  # pour ROI théorique

    for i, m in enumerate(matches):
        c = [m.get("cote_1", 0), m.get("cote_n", 0), m.get("cote_2", 0)]
        valid = [x for x in c if x > 0]
        if not valid:
            continue

        resultat = m.get("resultat", "")
        fav_idx = c.index(min(valid))
        fav_result = ["1", "N", "2"][fav_idx]

        if resultat == fav_result:
            nb_fav_gagne += 1
            # ROI : on mise 1 EUR sur le favori, on gagne cote_fav - 1
            roi_gains.append(min(valid) - 1)
        elif resultat:
            nb_surprises += 1
            # ROI : on perd la mise
            roi_gains.append(-1)

        # Cote du résultat réel
        result_map = {"1": 0, "N": 1, "2": 2}
        if resultat in result_map:
            cote_resultat = c[result_map[resultat]]
            if cote_resultat > 0:
                cotes_resultats.append(cote_resultat)

        cotes_fav_match.append(min(valid))

    f["nb_surprises"] = nb_surprises
    f["nb_fav_gagne"] = nb_fav_gagne
    f["pct_fav_gagne"] = nb_fav_gagne / nb_matchs if nb_matchs else 0

    # ROI théorique (mise 1 EUR sur chaque favori)
    f["roi_theorique"] = sum(roi_gains) / len(roi_gains) if roi_gains else 0

    if cotes_resultats:
        f["moy_cote_resultat"] = mean(cotes_resultats)
        f["somme_cotes_resultat"] = sum(cotes_resultats)
        f["prod_cotes_resultat"] = math.prod(cotes_resultats)
        f["max_cote_resultat"] = max(cotes_resultats)
        f["min_cote_resultat"] = min(cotes_resultats)
        f["med_cote_resultat"] = median(cotes_resultats)
    else:
        f["moy_cote_resultat"] = 0
        f["somme_cotes_resultat"] = 0
        f["prod_cotes_resultat"] = 0
        f["max_cote_resultat"] = 0
        f["min_cote_resultat"] = 0
        f["med_cote_resultat"] = 0

    # Surprise index : produit des cotes des résultats réels
    f["surprise_index"] = f["prod_cotes_resultat"]
    f["log_surprise_index"] = math.log(f["prod_cotes_resultat"]) if f["prod_cotes_resultat"] > 0 else 0

    # Brier score réalisé : score de calibration post-hoc
    brier_realise_values = []
    for m in matches:
        c = [m.get("cote_1", 0), m.get("cote_n", 0), m.get("cote_2", 0)]
        valid = [x for x in c if x > 0]
        resultat = m.get("resultat", "")
        if valid and resultat in ("1", "N", "2"):
            probs_impl = [1 / x if x > 0 else 0 for x in c]
            total_prob = sum(probs_impl)
            if total_prob > 0:
                probs_norm = [p / total_prob for p in probs_impl]
                result_map = {"1": 0, "N": 1, "2": 2}
                r_idx = result_map[resultat]
                indicators = [1 if i == r_idx else 0 for i in range(3)]
                brier = sum((probs_norm[i] - indicators[i]) ** 2 for i in range(3))
                brier_realise_values.append(brier)
    f["brier_realise"] = mean(brier_realise_values) if brier_realise_values else 0

    # Cyborg accuracy
    nb_cyborg_correct = 0
    nb_cyborg_total = 0
    for m in matches:
        prono = m.get("prono_cyborg", "")
        resultat = m.get("resultat", "")
        if prono and resultat and prono not in ("-", ""):
            nb_cyborg_total += 1
            if resultat in prono:
                nb_cyborg_correct += 1
    f["cyborg_accuracy"] = nb_cyborg_correct / nb_cyborg_total if nb_cyborg_total > 0 else 0
    f["nb_cyborg_correct"] = nb_cyborg_correct

    # Accord joueurs-résultat
    nb_joueurs_correct = 0
    for m in matches:
        p = [m.get("pct_1", 0), m.get("pct_n", 0), m.get("pct_2", 0)]
        resultat = m.get("resultat", "")
        if sum(p) > 0 and resultat:
            plus_joue_idx = p.index(max(p))
            plus_joue = ["1", "N", "2"][plus_joue_idx]
            if plus_joue == resultat:
                nb_joueurs_correct += 1
    f["joueurs_accuracy"] = nb_joueurs_correct / nb_matchs if nb_matchs else 0

    # Rapports (gains) — clés génériques
    rapports = grid.get("rapports", {})
    r_parfait = rapports.get(f"{nb_matchs}_sur_{nb_matchs}", {})
    r_presque = rapports.get(f"{nb_matchs - 1}_sur_{nb_matchs}", {})
    f["gagnants_parfait"] = r_parfait.get("gagnants", 0) or 0
    f["montant_parfait"] = r_parfait.get("montant", 0) or 0
    f["gagnants_presque"] = r_presque.get("gagnants", 0) or 0
    f["montant_presque"] = r_presque.get("montant", 0) or 0
    f["has_gagnant_parfait"] = 1 if f["gagnants_parfait"] > 0 else 0

    # Entropie du résultat
    if resultats:
        total_r = len(resultats)
        freq = {c: resultats.count(c) / total_r for c in "1N2"}
        f["entropie_resultat"] = -sum(
            freq[c] * math.log2(freq[c]) if freq[c] > 0 else 0 for c in "1N2"
        )
    else:
        f["entropie_resultat"] = 0

    # Alternance dans les résultats
    alternances = sum(1 for i in range(len(resultats) - 1) if resultats[i] != resultats[i + 1])
    f["alternance"] = alternances / (len(resultats) - 1) if len(resultats) > 1 else 0

    # Plus longue suite identique
    max_suite = 1
    current = 1
    for i in range(1, len(resultats)):
        if resultats[i] == resultats[i - 1]:
            current += 1
            max_suite = max(max_suite, current)
        else:
            current = 1
    f["plus_longue_suite"] = max_suite

    # --- INDICATEURS CROISÉS FARFELUS ---

    f["indice_chaos"] = f["nb_surprises"] * f["entropie_resultat"] * f["alternance"]
    f["indice_previsibilite"] = f["pct_fav_gagne"] * f["accord_cotes_joueurs"] * f["cyborg_accuracy"]
    f["ratio_resultat_fav"] = f["moy_cote_resultat"] / f["moy_cote_fav"] if f["moy_cote_fav"] > 0 else 0
    f["log_prod_cotes_resultat"] = math.log(f["prod_cotes_resultat"]) if f["prod_cotes_resultat"] > 0 else 0
    f["diff_x_surprises"] = f["difficulty"] * f["nb_surprises"]
    f["inv_somme_prob_fav"] = 1 / f["somme_prob_fav"] if f["somme_prob_fav"] > 0 else 0

    # Score de "tension"
    if pcts_fav and pcts_outsider:
        tensions = [abs(pf - po) for pf, po in zip(pcts_fav, pcts_outsider)]
        f["tension_moy"] = mean(tensions)
        f["tension_max"] = max(tensions)
    else:
        f["tension_moy"] = 0
        f["tension_max"] = 0

    # Pool estimé
    f["pool_estime_parfait"] = f["montant_parfait"] * f["gagnants_parfait"]

    return f


# =========================================================================
# 2. Calcul des corrélations
# =========================================================================

def compute_correlations(features_list: list[dict]) -> list[dict]:
    """Calcule toutes les corrélations entre les features pré et post match."""

    pre_match_keys = [
        "difficulty", "nb_pronostics",
        "moy_cote_fav", "med_cote_fav", "min_cote_fav", "max_cote_fav",
        "std_cote_fav", "somme_cotes_fav", "prod_cotes_fav",
        "moy_cote_outsider", "max_cote_outsider", "somme_cotes_outsider",
        "moy_spread", "max_spread", "min_spread", "std_spread",
        "moy_cote_nul", "std_cote_nul",
        "ratio_fav_outsider", "ratio_nul_fav",
        "diff_x_std", "prod_min_max_fav", "range_cote_fav", "coeff_variation_fav",
        "moy_prob_fav", "min_prob_fav", "max_prob_fav", "somme_prob_fav", "entropie_cotes",
        "moy_pct_fav", "min_pct_fav", "max_pct_fav", "moy_pct_outsider",
        "accord_cotes_joueurs", "ecart_pct_fav_outsider",
        "moy_dispersion_votes", "max_dispersion_votes", "min_dispersion_votes",
        "nb_cyborg_simple", "nb_cyborg_double", "ratio_cyborg_simple",
        "nb_matchs_serres", "nb_matchs_faciles", "nb_matchs_tres_serres",
        "inv_spread_sum", "harmonic_mean_fav",
        "pos_match_plus_serre", "pos_match_plus_facile",
        "asymetrie_cotes", "gradient_difficulte",
        # Nouvelles métriques pré-match
        "hhi_moyen", "hhi_max", "hhi_min",
        "gini_cotes_fav", "skewness_cotes_fav", "kurtosis_cotes_fav",
        "concentration_top3", "kelly_moyen", "brier_ante",
        "marge_bookmaker_moy",
    ]

    post_match_keys = [
        "nb_1", "nb_N", "nb_2",
        "nb_surprises", "nb_fav_gagne", "pct_fav_gagne",
        "moy_cote_resultat", "somme_cotes_resultat", "prod_cotes_resultat",
        "max_cote_resultat", "min_cote_resultat", "med_cote_resultat",
        "cyborg_accuracy", "nb_cyborg_correct",
        "joueurs_accuracy",
        "gagnants_parfait", "montant_parfait", "gagnants_presque", "montant_presque",
        "has_gagnant_parfait",
        "entropie_resultat", "alternance", "plus_longue_suite",
        "indice_chaos", "indice_previsibilite",
        "ratio_resultat_fav", "log_prod_cotes_resultat",
        "diff_x_surprises",
        "pool_estime_parfait",
        # Nouvelles métriques post-match
        "roi_theorique", "surprise_index", "log_surprise_index",
        "brier_realise",
    ]

    results = []

    for pre_key in pre_match_keys:
        for post_key in post_match_keys:
            x = [f[pre_key] for f in features_list if pre_key in f and post_key in f]
            y = [f[post_key] for f in features_list if pre_key in f and post_key in f]

            if len(x) < 10:
                continue

            if len(set(x)) < 3 or len(set(y)) < 3:
                continue

            try:
                r = correlation(x, y)
            except (ValueError, ZeroDivisionError):
                continue

            x_arr = np.array(x)
            y_arr = np.array(y)
            slope, intercept = np.polyfit(x_arr, y_arr, 1)

            r_squared = r ** 2

            y_pred = slope * x_arr + intercept
            residuals = y_arr - y_pred
            std_res = np.std(residuals) if len(residuals) > 1 else 1
            in_band = np.sum(np.abs(residuals) < std_res)
            pct_in_band = in_band / len(y_arr) * 100

            in_tight = np.sum(np.abs(residuals) < 0.5 * std_res)
            pct_in_tight = in_tight / len(y_arr) * 100

            results.append({
                "pre": pre_key,
                "post": post_key,
                "r": round(r, 4),
                "r2": round(r_squared, 4),
                "slope": round(slope, 6),
                "intercept": round(intercept, 4),
                "n": len(x),
                "pct_in_1std": round(pct_in_band, 1),
                "pct_in_0.5std": round(pct_in_tight, 1),
                "std_residuals": round(std_res, 4),
            })

    return results


# =========================================================================
# 3. Corrélations entre pré-match (combinaisons de 2 features)
# =========================================================================

def compute_combo_correlations(features_list: list[dict], top_n: int = 50) -> list[dict]:
    """Cherche des ratios/produits de 2 features pré-match corrélés aux post-match."""

    pre_keys_short = [
        "difficulty", "moy_cote_fav", "std_cote_fav", "moy_spread",
        "moy_prob_fav", "moy_pct_fav", "accord_cotes_joueurs",
        "moy_dispersion_votes", "nb_matchs_serres", "nb_matchs_faciles",
        "inv_spread_sum", "coeff_variation_fav", "ratio_nul_fav",
        "asymetrie_cotes", "gradient_difficulte", "nb_pronostics",
        "ratio_fav_outsider", "harmonic_mean_fav",
        "hhi_moyen", "gini_cotes_fav", "kelly_moyen", "marge_bookmaker_moy",
    ]

    post_keys_short = [
        "nb_surprises", "pct_fav_gagne", "montant_parfait",
        "gagnants_parfait", "has_gagnant_parfait",
        "moy_cote_resultat", "alternance",
        "cyborg_accuracy", "joueurs_accuracy",
        "roi_theorique", "brier_realise",
    ]

    results = []

    for k1, k2 in combinations(pre_keys_short, 2):
        for post_key in post_keys_short:
            # Essayer ratio k1/k2
            x_ratio = []
            y_vals = []
            for f in features_list:
                if k1 in f and k2 in f and post_key in f:
                    denom = f[k2]
                    if denom != 0:
                        x_ratio.append(f[k1] / denom)
                        y_vals.append(f[post_key])

            if len(x_ratio) >= 10 and len(set(x_ratio)) >= 3 and len(set(y_vals)) >= 3:
                try:
                    r = correlation(x_ratio, y_vals)
                    if abs(r) > 0.35:
                        results.append({
                            "combo": f"{k1}/{k2}",
                            "post": post_key,
                            "r": round(r, 4),
                            "r2": round(r ** 2, 4),
                            "n": len(x_ratio),
                            "type": "ratio",
                        })
                except (ValueError, ZeroDivisionError):
                    pass

            # Essayer produit k1 * k2
            x_prod = []
            y_vals2 = []
            for f in features_list:
                if k1 in f and k2 in f and post_key in f:
                    x_prod.append(f[k1] * f[k2])
                    y_vals2.append(f[post_key])

            if len(x_prod) >= 10 and len(set(x_prod)) >= 3 and len(set(y_vals2)) >= 3:
                try:
                    r = correlation(x_prod, y_vals2)
                    if abs(r) > 0.35:
                        results.append({
                            "combo": f"{k1}*{k2}",
                            "post": post_key,
                            "r": round(r, 4),
                            "r2": round(r ** 2, 4),
                            "n": len(x_prod),
                            "type": "produit",
                        })
                except (ValueError, ZeroDivisionError):
                    pass

    results.sort(key=lambda x: abs(x["r"]), reverse=True)
    return results[:top_n]


# =========================================================================
# 4. Analyse des profils (patterns 1N2)
# =========================================================================

def analyze_profiles(features_list: list[dict]) -> dict:
    """Analyse les profils 1N2 et leurs corrélations avec les cotes."""
    profil_counts = Counter()
    profil_features = {}

    for f in features_list:
        profil = f.get("profil", "")
        if not profil:
            continue
        profil_counts[profil] += 1
        if profil not in profil_features:
            profil_features[profil] = []
        profil_features[profil].append(f)

    analysis = {
        "distribution_profils": dict(profil_counts.most_common()),
        "profils_detail": {},
    }

    for profil, flist in profil_features.items():
        if len(flist) < 2:
            continue
        analysis["profils_detail"][profil] = {
            "count": len(flist),
            "moy_difficulty": round(mean(f["difficulty"] for f in flist), 2),
            "moy_cote_fav": round(mean(f["moy_cote_fav"] for f in flist), 2),
            "moy_nb_surprises": round(mean(f["nb_surprises"] for f in flist), 2),
            "moy_montant_parfait": round(mean(f["montant_parfait"] for f in flist), 2),
            "moy_gagnants_parfait": round(mean(f["gagnants_parfait"] for f in flist), 1),
            "moy_pct_fav_gagne": round(mean(f["pct_fav_gagne"] for f in flist), 3),
            "moy_accord_cotes_joueurs": round(mean(f["accord_cotes_joueurs"] for f in flist), 3),
        }

    return analysis


# =========================================================================
# 5. Détection de seuils (binning)
# =========================================================================

def detect_thresholds(features_list: list[dict]) -> list[dict]:
    """Cherche des seuils sur les features pré-match qui séparent bien les résultats."""

    threshold_keys = [
        "difficulty", "moy_cote_fav", "std_cote_fav", "moy_spread",
        "moy_prob_fav", "accord_cotes_joueurs", "moy_dispersion_votes",
        "nb_matchs_serres", "nb_matchs_faciles", "coeff_variation_fav",
        "inv_spread_sum", "ratio_nul_fav", "nb_pronostics",
        "ratio_fav_outsider",
        "hhi_moyen", "gini_cotes_fav", "kelly_moyen", "marge_bookmaker_moy",
    ]

    target_keys = [
        "nb_surprises", "pct_fav_gagne", "montant_parfait",
        "has_gagnant_parfait", "moy_cote_resultat",
        "roi_theorique", "brier_realise",
    ]

    results = []

    for pre_key in threshold_keys:
        values = sorted(set(f[pre_key] for f in features_list if pre_key in f))
        if len(values) < 5:
            continue

        for target_key in target_keys:
            best_split = None
            best_diff = 0

            for q in [0.25, 0.33, 0.5, 0.67, 0.75]:
                threshold = np.quantile(
                    [f[pre_key] for f in features_list], q
                )

                below = [f[target_key] for f in features_list if f[pre_key] <= threshold]
                above = [f[target_key] for f in features_list if f[pre_key] > threshold]

                if len(below) < 5 or len(above) < 5:
                    continue

                mean_below = mean(below)
                mean_above = mean(above)
                diff = abs(mean_above - mean_below)

                all_vals = below + above
                global_std = stdev(all_vals) if len(all_vals) > 1 else 1
                effect_size = diff / global_std if global_std > 0 else 0

                if effect_size > best_diff:
                    best_diff = effect_size
                    best_split = {
                        "pre": pre_key,
                        "target": target_key,
                        "threshold": round(threshold, 3),
                        "quantile": q,
                        "mean_below": round(mean_below, 3),
                        "mean_above": round(mean_above, 3),
                        "n_below": len(below),
                        "n_above": len(above),
                        "effect_size": round(effect_size, 3),
                        "diff_pct": round(diff / abs(mean_below) * 100, 1) if mean_below != 0 else 0,
                    }

            if best_split and best_split["effect_size"] > 0.4:
                results.append(best_split)

    results.sort(key=lambda x: x["effect_size"], reverse=True)
    return results


# =========================================================================
# 6. Rapport final
# =========================================================================

def generate_report(grid_type: str = "LF7", history_path: str = None) -> str:
    """Génère un rapport complet d'analyse des corrélations pour un type donné."""

    history = load_history(path=history_path, grid_type=grid_type)
    if not history:
        return f"ERREUR: Aucun historique trouvé pour {grid_type}."

    # Extraction
    features_list = []
    for grid in history:
        feat = extract_grid_features(grid)
        if feat:
            features_list.append(feat)

    if len(features_list) < 10:
        return f"Pas assez de grilles ({len(features_list)}) pour une analyse significative."

    nb_matchs = features_list[0].get("nb_matchs", "?")

    lines = []
    lines.append("=" * 80)
    lines.append(f"ANALYSE DE CORRÉLATIONS AVANT/APRÈS MATCH — {grid_type} ({nb_matchs} matchs)")
    lines.append(f"Nombre de grilles analysées: {len(features_list)}")
    lines.append("=" * 80)

    # --- Stats descriptives ---
    lines.append("\n" + "=" * 80)
    lines.append("1. STATISTIQUES DESCRIPTIVES")
    lines.append("=" * 80)

    desc_keys = [
        ("difficulty", "Difficulté"),
        ("moy_cote_fav", "Moy. cote favori"),
        ("std_cote_fav", "Écart-type cote favori"),
        ("nb_surprises", "Nb surprises"),
        ("pct_fav_gagne", "% favori gagne"),
        ("montant_parfait", f"Montant {nb_matchs}/{nb_matchs}"),
        ("gagnants_parfait", f"Gagnants {nb_matchs}/{nb_matchs}"),
        ("montant_presque", f"Montant {nb_matchs-1}/{nb_matchs}"),
        ("moy_cote_resultat", "Moy. cote du résultat"),
        ("accord_cotes_joueurs", "Accord cotes/joueurs"),
        ("cyborg_accuracy", "Accuracy Cyborg"),
        ("joueurs_accuracy", "Accuracy joueurs"),
        ("alternance", "Alternance"),
        ("entropie_resultat", "Entropie résultat"),
        ("hhi_moyen", "HHI moyen"),
        ("gini_cotes_fav", "Gini cotes favoris"),
        ("skewness_cotes_fav", "Skewness cotes favoris"),
        ("kelly_moyen", "Kelly fractionnel moyen"),
        ("marge_bookmaker_moy", "Marge bookmaker moy."),
        ("roi_theorique", "ROI théorique"),
        ("brier_realise", "Brier score réalisé"),
        ("brier_ante", "Brier score ex-ante"),
    ]

    for key, label in desc_keys:
        vals = [feat[key] for feat in features_list if key in feat and feat[key] is not None]
        if vals:
            lines.append(
                f"  {label:30s}: moy={mean(vals):8.3f}  "
                f"med={median(vals):8.3f}  "
                f"min={min(vals):8.3f}  max={max(vals):8.3f}  "
                f"std={stdev(vals) if len(vals) > 1 else 0:8.3f}"
            )

    # --- Corrélations simples ---
    lines.append("\n" + "=" * 80)
    lines.append("2. TOP CORRÉLATIONS SIMPLES (|r| > 0.25)")
    lines.append("=" * 80)

    corrs = compute_correlations(features_list)
    corrs.sort(key=lambda x: abs(x["r"]), reverse=True)

    strong = [c for c in corrs if abs(c["r"]) > 0.25]
    for c in strong[:60]:
        direction = "+" if c["r"] > 0 else "-"
        lines.append(
            f"  {direction} r={c['r']:+.4f}  R2={c['r2']:.4f}  "
            f"{c['pct_in_1std']:5.1f}% in +/-1std  "
            f"{c['pre']:30s} -> {c['post']}"
        )

    # --- Corrélations combinées ---
    lines.append("\n" + "=" * 80)
    lines.append("3. TOP CORRÉLATIONS COMBINÉES (ratios & produits, |r| > 0.35)")
    lines.append("=" * 80)

    combos = compute_combo_correlations(features_list)
    for c in combos[:30]:
        direction = "+" if c["r"] > 0 else "-"
        lines.append(
            f"  {direction} r={c['r']:+.4f}  R2={c['r2']:.4f}  "
            f"[{c['type']:7s}] {c['combo']:40s} -> {c['post']}"
        )

    # --- Profils ---
    lines.append("\n" + "=" * 80)
    lines.append("4. ANALYSE DES PROFILS 1N2")
    lines.append("=" * 80)

    profiles = analyze_profiles(features_list)
    lines.append(f"\n  Distribution: {profiles['distribution_profils']}")

    for profil, detail in sorted(
        profiles["profils_detail"].items(),
        key=lambda x: x[1]["count"], reverse=True,
    ):
        lines.append(
            f"\n  Profil {profil} (n={detail['count']}): "
            f"diff={detail['moy_difficulty']:.1f}  "
            f"cote_fav={detail['moy_cote_fav']:.2f}  "
            f"surprises={detail['moy_nb_surprises']:.1f}  "
            f"%fav={detail['moy_pct_fav_gagne']:.1%}  "
            f"montant={detail['moy_montant_parfait']:.0f}EUR  "
            f"gagnants={detail['moy_gagnants_parfait']:.0f}"
        )

    # --- Seuils ---
    lines.append("\n" + "=" * 80)
    lines.append("5. SEUILS DISCRIMINANTS (effect size > 0.4)")
    lines.append("=" * 80)

    thresholds = detect_thresholds(features_list)
    for t in thresholds[:30]:
        lines.append(
            f"  {t['pre']:30s} seuil={t['threshold']:8.3f} (Q{t['quantile']:.0%})"
            f"  -> {t['target']:20s}: "
            f"bas={t['mean_below']:.3f} (n={t['n_below']}) vs "
            f"haut={t['mean_above']:.3f} (n={t['n_above']})  "
            f"effect={t['effect_size']:.3f}  diff={t['diff_pct']:+.1f}%"
        )

    # --- Résumé des découvertes ---
    lines.append("\n" + "=" * 80)
    lines.append("6. RÉSUMÉ DES DÉCOUVERTES CLÉS")
    lines.append("=" * 80)

    if strong:
        top3 = strong[:3]
        lines.append("\n  Top 3 corrélations les plus fortes:")
        for i, c in enumerate(top3, 1):
            lines.append(
                f"    {i}. {c['pre']} -> {c['post']}: r={c['r']:+.4f} "
                f"({c['pct_in_1std']:.0f}% des points dans +/-1std)"
            )

    if combos:
        lines.append("\n  Top 3 combinaisons farfelues:")
        for i, c in enumerate(combos[:3], 1):
            lines.append(
                f"    {i}. [{c['type']}] {c['combo']} -> {c['post']}: r={c['r']:+.4f}"
            )

    if thresholds:
        lines.append("\n  Top 3 seuils les plus discriminants:")
        for i, t in enumerate(thresholds[:3], 1):
            lines.append(
                f"    {i}. {t['pre']} {'<' if t['mean_below'] < t['mean_above'] else '>'} "
                f"{t['threshold']:.2f} -> {t['target']}: "
                f"{t['mean_below']:.3f} vs {t['mean_above']:.3f} "
                f"(effect size={t['effect_size']:.2f})"
            )

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    grid_type = sys.argv[1] if len(sys.argv) > 1 else "LF7"
    report = generate_report(grid_type=grid_type)
    print(report)

    # Sauvegarder
    report_path = os.path.join("data", "history", f"correlation_report_{grid_type}.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fout:
        fout.write(report)
    print(f"\nRapport sauvegardé: {report_path}")
