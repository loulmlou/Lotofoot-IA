"""Baselines de prédiction pour comparaison avec les modèles ML."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


LABEL_MAP = {"1": 0, "N": 1, "2": 2}
LABEL_NAMES = ["1", "N", "2"]


def _encode_labels(y_true):
    """Convertit les labels textuels en entiers."""
    return np.array([LABEL_MAP[y] for y in y_true])


def evaluate_predictions(y_true, y_pred, y_proba=None, cotes=None) -> dict:
    """Evalue les prédictions avec accuracy, log-loss et ROI simulé.

    Args:
        y_true: labels réels (list de '1', 'N', '2')
        y_pred: labels prédits (list de '1', 'N', '2')
        y_proba: probabilités prédites (array Nx3, colonnes 1/N/2). Optionnel.
        cotes: DataFrame ou list de dict avec cote_1, cote_n, cote_2. Optionnel.

    Returns:
        dict avec accuracy, log_loss (si y_proba), roi (si cotes)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)

    result = {"accuracy": acc}

    # Log-loss
    if y_proba is not None:
        y_true_encoded = _encode_labels(y_true)
        # Clip pour éviter log(0)
        y_proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
        ll = log_loss(y_true_encoded, y_proba_clipped, labels=[0, 1, 2])
        result["log_loss"] = ll

    # ROI simulé (mise 1 sur chaque prédiction)
    if cotes is not None:
        if isinstance(cotes, pd.DataFrame):
            cotes_list = cotes.to_dict("records")
        else:
            cotes_list = list(cotes)

        total_mise = 0
        total_gains = 0.0

        for i in range(len(y_pred)):
            if i >= len(cotes_list):
                break
            c = cotes_list[i]
            pred = y_pred_arr[i]
            vrai = y_true_arr[i]

            cote_val = None
            if pred == "1":
                cote_val = c.get("cote_1")
            elif pred == "N":
                cote_val = c.get("cote_n")
            elif pred == "2":
                cote_val = c.get("cote_2")

            if cote_val and cote_val > 0:
                total_mise += 1
                if pred == vrai:
                    total_gains += cote_val

        roi = (total_gains - total_mise) / total_mise if total_mise > 0 else 0.0
        result["roi"] = roi
        result["total_mise"] = total_mise
        result["total_gains"] = total_gains

    return result


def baseline_random(df: pd.DataFrame) -> dict:
    """Baseline aléatoire : prédiction uniforme (33% chaque résultat).

    Args:
        df: DataFrame avec colonne 'resultat' et optionnellement cote_1/cote_n/cote_2
    """
    np.random.seed(42)
    y_true = df["resultat"].tolist()
    y_pred = np.random.choice(["1", "N", "2"], size=len(y_true)).tolist()
    y_proba = np.full((len(y_true), 3), 1.0 / 3.0)

    cotes = None
    if "cote_1" in df.columns and "cote_n" in df.columns and "cote_2" in df.columns:
        cotes = df[["cote_1", "cote_n", "cote_2"]].to_dict("records")

    result = evaluate_predictions(y_true, y_pred, y_proba=y_proba, cotes=cotes)
    result["name"] = "random"
    return result


def baseline_home(df: pd.DataFrame) -> dict:
    """Baseline toujours domicile : prédit toujours '1'.

    Args:
        df: DataFrame avec colonne 'resultat'
    """
    y_true = df["resultat"].tolist()
    y_pred = ["1"] * len(y_true)
    y_proba = np.zeros((len(y_true), 3))
    y_proba[:, 0] = 1.0  # 100% sur domicile

    cotes = None
    if "cote_1" in df.columns and "cote_n" in df.columns and "cote_2" in df.columns:
        cotes = df[["cote_1", "cote_n", "cote_2"]].to_dict("records")

    result = evaluate_predictions(y_true, y_pred, y_proba=y_proba, cotes=cotes)
    result["name"] = "always_home"
    return result


def baseline_odds_favorite(df: pd.DataFrame) -> dict:
    """Baseline favoris cotes : prédit toujours le résultat avec la plus petite cote.

    Args:
        df: DataFrame avec colonnes prob_1, prob_n, prob_2 (ou cote_1, cote_n, cote_2)
            et colonne 'resultat'
    """
    y_true = df["resultat"].tolist()
    y_pred = []
    y_proba = []

    for _, row in df.iterrows():
        # Utiliser les probabilités normalisées si disponibles
        if "prob_1" in df.columns and pd.notna(row.get("prob_1")):
            p1 = row["prob_1"]
            pn = row["prob_n"]
            p2 = row["prob_2"]
        elif "cote_1" in df.columns and pd.notna(row.get("cote_1")):
            c1 = row["cote_1"]
            cn = row["cote_n"]
            c2 = row["cote_2"]
            if c1 and cn and c2:
                raw = [1/c1, 1/cn, 1/c2]
                total = sum(raw)
                p1, pn, p2 = raw[0]/total, raw[1]/total, raw[2]/total
            else:
                p1, pn, p2 = 1/3, 1/3, 1/3
        else:
            p1, pn, p2 = 1/3, 1/3, 1/3

        probs = [p1, pn, p2]
        y_proba.append(probs)
        pred_idx = np.argmax(probs)
        y_pred.append(LABEL_NAMES[pred_idx])

    y_proba = np.array(y_proba)

    cotes = None
    if "cote_1" in df.columns and "cote_n" in df.columns and "cote_2" in df.columns:
        cotes = df[["cote_1", "cote_n", "cote_2"]].to_dict("records")

    result = evaluate_predictions(y_true, y_pred, y_proba=y_proba, cotes=cotes)
    result["name"] = "odds_favorite"
    return result


def baseline_distribution(df: pd.DataFrame) -> dict:
    """Baseline distribution : prédit selon la distribution historique.

    Distribution typique : ~44% dom, ~27% nul, ~29% ext.
    Utilise la distribution réelle du dataset.

    Args:
        df: DataFrame avec colonne 'resultat'
    """
    np.random.seed(42)
    y_true = df["resultat"].tolist()

    # Calculer la distribution réelle
    counts = pd.Series(y_true).value_counts(normalize=True)
    p1 = counts.get("1", 0.0)
    pn = counts.get("N", 0.0)
    p2 = counts.get("2", 0.0)

    y_pred = np.random.choice(
        ["1", "N", "2"], size=len(y_true), p=[p1, pn, p2]
    ).tolist()

    y_proba = np.full((len(y_true), 3), [p1, pn, p2])

    cotes = None
    if "cote_1" in df.columns and "cote_n" in df.columns and "cote_2" in df.columns:
        cotes = df[["cote_1", "cote_n", "cote_2"]].to_dict("records")

    result = evaluate_predictions(y_true, y_pred, y_proba=y_proba, cotes=cotes)
    result["name"] = "distribution"
    return result
