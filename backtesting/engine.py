"""Moteur de backtesting — simule les prédictions sur les matchs historiques
et évalue les performances en conditions de grilles Loto Foot."""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from analysis.baseline import LABEL_MAP, LABEL_NAMES, evaluate_predictions
from analysis.dataset_builder import load_dataset, split_by_date
from config.settings import SCORING_WEIGHTS, LOTOFOOT_TYPES
from models.predictor import Predictor, STRATEGY_THRESHOLDS
from models.scoring import compute_final_score, score_lotofoot
from models.train import load_model, DEFAULT_MODEL_DIR, FEATURE_COLS


def _load_artifacts():
    """Charge le modèle et le preprocessor depuis le disque."""
    model_path = os.path.join(DEFAULT_MODEL_DIR, "xgboost.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(DEFAULT_MODEL_DIR, "lightgbm.joblib")
    if not os.path.exists(model_path):
        return None, None, None

    artifact = load_model(model_path)
    return artifact["model"], artifact["preprocessor"], artifact["feature_cols"]


def _batch_ml_proba(df, model, preprocessor, feature_cols):
    """Calcule les probabilités ML en batch pour tout le DataFrame.

    Retourne un array (n, 3) de probabilités [p1, pn, p2].
    """
    import warnings

    available = [c for c in feature_cols if c in df.columns]
    X_raw = df[available].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = preprocessor.transform(X_raw)

    proba = model.predict_proba(X)
    return proba


def backtest_matches(df, model=None, preprocessor=None, feature_cols=None,
                     strategy="equilibree", grid_type=None):
    """Backteste le moteur de scoring sur un ensemble de matchs.

    Pour chaque match du DataFrame, calcule le score pondéré et compare
    la prédiction au résultat réel. Les prédictions ML sont calculées
    en batch pour la performance.

    Args:
        df: DataFrame avec features et 'resultat'
        model: modèle ML (optionnel)
        preprocessor: pipeline sklearn (optionnel)
        feature_cols: colonnes features (optionnel)
        strategy: stratégie de prédiction
        grid_type: type de grille Loto Foot

    Returns:
        dict avec métriques globales et DataFrame de prédictions détaillées
    """
    threshold = STRATEGY_THRESHOLDS.get(strategy, 0.10)

    # Pré-calculer les probabilités ML en batch
    ml_proba = None
    if model is not None and preprocessor is not None and feature_cols is not None:
        ml_proba = _batch_ml_proba(df, model, preprocessor, feature_cols)

    # Importer les fonctions de scoring individuelles pour éviter score_modele_ia row-by-row
    from models.scoring import (
        score_cotes, score_forme, score_classement, score_historique,
        score_lotofoot, score_contexte, _normalize, _apply_draw_correction,
    )

    weights = SCORING_WEIGHTS

    predictions = []
    y_true = []
    y_pred = []
    y_proba = []
    cotes_list = []

    for i, (_, row) in enumerate(df.iterrows()):
        features = row.to_dict()
        resultat = features.get("resultat")
        if resultat not in ("1", "N", "2"):
            continue

        # Calculer les scores composants (sauf modele_ia, fait en batch)
        scores = {
            "cotes": score_cotes(features),
            "forme": score_forme(features),
            "classement": score_classement(features),
            "historique": score_historique(features),
            "stats_lotofoot": score_lotofoot(features, grid_type=grid_type),
            "contexte": score_contexte(features),
        }

        # Ajouter le score IA depuis le batch
        if ml_proba is not None:
            scores["modele_ia"] = _normalize(ml_proba[i])
        else:
            scores["modele_ia"] = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        # Somme pondérée
        final = np.zeros(3)
        total_weight = 0.0
        for component, weight in weights.items():
            if component in scores:
                final += weight * scores[component]
                total_weight += weight
        if total_weight > 0:
            final = final / total_weight
        final = _apply_draw_correction(final)
        final = _normalize(final)

        pred_idx = int(np.argmax(final))
        prediction = LABEL_NAMES[pred_idx]
        sorted_probs = np.sort(final)[::-1]
        confiance = float(sorted_probs[0] - sorted_probs[1])

        filtre = confiance >= threshold

        predictions.append({
            "match_id": features.get("match_id"),
            "date": features.get("date"),
            "resultat_reel": resultat,
            "prediction": prediction,
            "correct": prediction == resultat,
            "prob_1": float(final[0]),
            "prob_n": float(final[1]),
            "prob_2": float(final[2]),
            "confiance": confiance,
            "filtre": filtre,
            "cote_1": features.get("cote_1"),
            "cote_n": features.get("cote_n"),
            "cote_2": features.get("cote_2"),
        })

        y_true.append(resultat)
        y_pred.append(prediction)
        y_proba.append([float(final[0]), float(final[1]), float(final[2])])

        cote_dict = {}
        for k in ("cote_1", "cote_n", "cote_2"):
            v = features.get(k)
            cote_dict[k] = v if v and not np.isnan(v) else None
        cotes_list.append(cote_dict)

    pred_df = pd.DataFrame(predictions)

    # Métriques globales (tous les matchs)
    metrics_all = evaluate_predictions(
        y_true, y_pred,
        y_proba=np.array(y_proba),
        cotes=cotes_list,
    )
    metrics_all["n_matchs"] = len(y_true)
    metrics_all["strategy"] = strategy

    # Métriques filtrées (seulement les matchs passant le seuil)
    filtered = pred_df[pred_df["filtre"]]
    metrics_filtered = {}
    if len(filtered) > 0:
        f_true = filtered["resultat_reel"].tolist()
        f_pred = filtered["prediction"].tolist()
        f_proba = filtered[["prob_1", "prob_n", "prob_2"]].values
        f_cotes = filtered[["cote_1", "cote_n", "cote_2"]].to_dict("records")
        metrics_filtered = evaluate_predictions(f_true, f_pred, y_proba=f_proba, cotes=f_cotes)
        metrics_filtered["n_matchs"] = len(filtered)
        metrics_filtered["pct_filtres"] = len(filtered) / len(pred_df) * 100
    else:
        metrics_filtered["n_matchs"] = 0
        metrics_filtered["accuracy"] = 0.0

    return {
        "all": metrics_all,
        "filtered": metrics_filtered,
        "predictions": pred_df,
    }


def simulate_grids(pred_df, grid_size=7, n_simulations=None):
    """Simule des grilles Loto Foot à partir des prédictions de matchs.

    Regroupe les matchs consécutifs (par date) en grilles de `grid_size`
    matchs et calcule le taux de grilles parfaites (tous les résultats corrects).

    Args:
        pred_df: DataFrame de prédictions (sortie de backtest_matches)
        grid_size: nombre de matchs par grille (7, 8, 12 ou 15)
        n_simulations: nombre max de grilles à simuler (None = toutes)

    Returns:
        dict avec statistiques des grilles simulées
    """
    df = pred_df.sort_values("date").reset_index(drop=True)

    n_grids = len(df) // grid_size
    if n_simulations:
        n_grids = min(n_grids, n_simulations)

    if n_grids == 0:
        return {"n_grilles": 0, "message": "Pas assez de matchs pour simuler"}

    grids = []
    for i in range(n_grids):
        start = i * grid_size
        end = start + grid_size
        grid_df = df.iloc[start:end]

        n_correct = grid_df["correct"].sum()
        all_correct = n_correct == grid_size

        # Résultat simulé vs résultat réel
        pred_str = "".join(grid_df["prediction"].tolist())
        real_str = "".join(grid_df["resultat_reel"].tolist())

        # Confiance moyenne de la grille
        avg_conf = grid_df["confiance"].mean()

        # ROI simulé si on pariait sur la grille
        gains = 0.0
        for _, row in grid_df.iterrows():
            if row["correct"]:
                cote_key = f"cote_{row['prediction'].lower()}" if row["prediction"] != "N" else "cote_n"
                if row["prediction"] == "1":
                    cote_key = "cote_1"
                elif row["prediction"] == "2":
                    cote_key = "cote_2"
                else:
                    cote_key = "cote_n"
                cote_val = row.get(cote_key)
                if cote_val and not np.isnan(cote_val):
                    gains += cote_val

        grids.append({
            "grille_num": i + 1,
            "n_correct": int(n_correct),
            "all_correct": all_correct,
            "prediction": pred_str,
            "reel": real_str,
            "confiance_moy": avg_conf,
            "gains_paris": gains,
            "date_debut": grid_df["date"].iloc[0],
            "date_fin": grid_df["date"].iloc[-1],
        })

    grids_df = pd.DataFrame(grids)

    # Statistiques
    n_parfaites = grids_df["all_correct"].sum()
    distribution_correct = grids_df["n_correct"].value_counts().sort_index()

    return {
        "n_grilles": n_grids,
        "grid_size": grid_size,
        "n_parfaites": int(n_parfaites),
        "pct_parfaites": n_parfaites / n_grids * 100 if n_grids > 0 else 0,
        "moy_correct": float(grids_df["n_correct"].mean()),
        "max_correct": int(grids_df["n_correct"].max()),
        "min_correct": int(grids_df["n_correct"].min()),
        "confiance_moy": float(grids_df["confiance_moy"].mean()),
        "distribution_correct": distribution_correct.to_dict(),
        "grilles": grids_df,
    }


def backtest_historical_grids(df, model=None, preprocessor=None, feature_cols=None):
    """Backteste les distributions de grilles historiques.

    Compare la distribution 1/N/2 des prédictions avec celle observée
    dans les vraies grilles Loto Foot.

    Args:
        df: DataFrame de matchs avec features

    Returns:
        dict avec analyse des distributions
    """
    from analysis.grid_analysis import get_distribution_by_type

    try:
        real_dist = get_distribution_by_type()
    except Exception:
        real_dist = {}

    # Distribution des prédictions du modèle
    pred_results = backtest_matches(df, model, preprocessor, feature_cols)
    pred_df = pred_results["predictions"]

    pred_dist = pred_df["prediction"].value_counts(normalize=True)
    real_match_dist = pred_df["resultat_reel"].value_counts(normalize=True)

    analysis = {
        "prediction_distribution": {
            "1": float(pred_dist.get("1", 0)),
            "N": float(pred_dist.get("N", 0)),
            "2": float(pred_dist.get("2", 0)),
        },
        "real_distribution": {
            "1": float(real_match_dist.get("1", 0)),
            "N": float(real_match_dist.get("N", 0)),
            "2": float(real_match_dist.get("2", 0)),
        },
        "grid_distributions": {},
    }

    for grid_type, info in real_dist.items():
        n = info["nb_grilles"]
        nb_matchs_type = {"LF7": 7, "LF8": 8, "LF12": 12, "LF15": 15}.get(grid_type, 7)
        analysis["grid_distributions"][grid_type] = {
            "nb_grilles": n,
            "moy_pct_1": info["moy_1"] / nb_matchs_type * 100,
            "moy_pct_n": info["moy_n"] / nb_matchs_type * 100,
            "moy_pct_2": info["moy_2"] / nb_matchs_type * 100,
        }

    return analysis


def run_full_backtest(dataset_path=None):
    """Lance le backtesting complet : matchs + grilles simulées + stratégies.

    Args:
        dataset_path: chemin vers le dataset CSV (optionnel)

    Returns:
        dict avec tous les résultats
    """
    print("=" * 70)
    print("BACKTESTING COMPLET")
    print("=" * 70)

    # 1. Charger les données
    if dataset_path is None:
        dataset_path = os.path.join("data", "dataset.csv")

    print(f"\nChargement du dataset: {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"  {len(df)} matchs chargés")

    # 2. Charger le modèle
    model, preprocessor, feature_cols = _load_artifacts()
    model_name = "aucun"
    if model is not None:
        model_name = type(model).__name__
    print(f"  Modèle: {model_name}")

    # 3. Split: backtester uniquement sur les données de test (post-2025)
    _, _, test_df = split_by_date(df, "2024-01-01", "2025-01-01")
    print(f"  Matchs de test (post-2025): {len(test_df)}")

    if len(test_df) < 50:
        # Fallback si pas assez de données post-2025
        print("  Pas assez de données post-2025, utilisation des 20% les plus récents...")
        n = len(df)
        test_df = df.iloc[int(n * 0.8):].copy()
        print(f"  Matchs de test: {len(test_df)}")

    results = {}

    # 4. Backtesting par stratégie
    print("\n" + "-" * 70)
    print("BACKTESTING PAR STRATÉGIE")
    print("-" * 70)

    for strategy in ["prudente", "equilibree", "audacieuse"]:
        print(f"\n--- Stratégie: {strategy.upper()} ---")
        bt = backtest_matches(
            test_df, model, preprocessor, feature_cols, strategy=strategy,
        )

        all_m = bt["all"]
        filt_m = bt["filtered"]

        print(f"  Tous les matchs ({all_m['n_matchs']}):")
        print(f"    Accuracy: {all_m['accuracy']:.4f}")
        print(f"    Log-loss: {all_m.get('log_loss', 'N/A')}")
        if "roi" in all_m:
            print(f"    ROI: {all_m['roi']:.4f}")

        if filt_m["n_matchs"] > 0:
            print(f"  Matchs filtrés ({filt_m['n_matchs']} — {filt_m.get('pct_filtres', 0):.1f}%):")
            print(f"    Accuracy: {filt_m['accuracy']:.4f}")
            if "roi" in filt_m:
                print(f"    ROI: {filt_m['roi']:.4f}")

        results[strategy] = bt

    # 5. Simulation de grilles
    print("\n" + "-" * 70)
    print("SIMULATION DE GRILLES LOTO FOOT")
    print("-" * 70)

    # Utiliser les prédictions de la stratégie équilibrée
    pred_df = results["equilibree"]["predictions"]

    grid_results = {}
    for grid_type, config in LOTOFOOT_TYPES.items():
        nb = config["nb_matchs"]
        code = config["code"]
        sim = simulate_grids(pred_df, grid_size=nb)

        if sim["n_grilles"] == 0:
            continue

        grid_results[code] = sim
        print(f"\n  {code} ({nb} matchs/grille) — {sim['n_grilles']} grilles simulées:")
        print(f"    Moyenne correct: {sim['moy_correct']:.2f}/{nb} "
              f"({sim['moy_correct']/nb*100:.1f}%)")
        print(f"    Min/Max correct: {sim['min_correct']}/{sim['max_correct']}")
        print(f"    Grilles parfaites: {sim['n_parfaites']} "
              f"({sim['pct_parfaites']:.2f}%)")
        print(f"    Distribution:")
        for n_ok, count in sorted(sim["distribution_correct"].items()):
            bar = "#" * min(count, 60)
            print(f"      {n_ok:2d}/{nb}: {count:4d} {bar}")

    results["grilles"] = grid_results

    # 6. Analyse des distributions
    print("\n" + "-" * 70)
    print("ANALYSE DES DISTRIBUTIONS")
    print("-" * 70)

    dist_analysis = backtest_historical_grids(test_df, model, preprocessor, feature_cols)
    results["distributions"] = dist_analysis

    print(f"\n  Distribution des prédictions:")
    pd_dist = dist_analysis["prediction_distribution"]
    print(f"    1: {pd_dist['1']*100:.1f}%  N: {pd_dist['N']*100:.1f}%  "
          f"2: {pd_dist['2']*100:.1f}%")

    print(f"  Distribution réelle (test):")
    rd_dist = dist_analysis["real_distribution"]
    print(f"    1: {rd_dist['1']*100:.1f}%  N: {rd_dist['N']*100:.1f}%  "
          f"2: {rd_dist['2']*100:.1f}%")

    if dist_analysis["grid_distributions"]:
        print(f"\n  Distribution moyenne des grilles historiques:")
        for gt, gd in dist_analysis["grid_distributions"].items():
            print(f"    {gt} ({gd['nb_grilles']} grilles): "
                  f"1={gd['moy_pct_1']:.1f}%  N={gd['moy_pct_n']:.1f}%  "
                  f"2={gd['moy_pct_2']:.1f}%")

    # 7. Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    eq = results["equilibree"]["all"]
    pr = results["prudente"]
    print(f"\n  Scoring pondéré (équilibrée): "
          f"accuracy={eq['accuracy']:.4f}, "
          f"log-loss={eq.get('log_loss', 'N/A')}")

    if pr["filtered"]["n_matchs"] > 0:
        print(f"  Filtrage prudent: "
              f"accuracy={pr['filtered']['accuracy']:.4f} "
              f"sur {pr['filtered']['n_matchs']} matchs "
              f"({pr['filtered'].get('pct_filtres', 0):.1f}%)")

    if "LF7" in grid_results:
        lf7 = grid_results["LF7"]
        print(f"  Grilles LF7: {lf7['moy_correct']:.2f}/7 correct en moyenne, "
              f"{lf7['n_parfaites']} parfaites sur {lf7['n_grilles']}")

    return results
