"""Construction du dataset d'entraînement pour les modèles ML."""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import select, func, and_, or_, text

from database.models import Match, Cote, Competition
from database.connection import SessionLocal, engine
from analysis.features import build_match_features


def _load_all_matches_df() -> pd.DataFrame:
    """Charge tous les matchs avec cotes en un seul DataFrame pandas."""
    query = """
        SELECT
            m.id AS match_id,
            m.date,
            m.equipe_dom_id,
            m.equipe_ext_id,
            m.competition_id,
            m.score_dom,
            m.score_ext,
            m.resultat,
            m.saison,
            m.journee,
            c.cote_1,
            c.cote_n,
            c.cote_2,
            comp.nom AS competition_nom
        FROM matchs m
        LEFT JOIN cotes c ON c.match_id = m.id
        LEFT JOIN competitions comp ON comp.id = m.competition_id
        WHERE m.resultat IS NOT NULL
        ORDER BY m.date
    """
    df = pd.read_sql(query, engine, parse_dates=["date"])
    return df


def _rolling_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Calcule la forme récente pour chaque équipe à chaque match (vectorisé).

    Retourne un DataFrame indexé par match_id avec les features de forme
    pour dom et ext.
    """
    # Créer une vue par équipe (dom + ext empilées)
    dom_view = df[["match_id", "date", "equipe_dom_id", "score_dom", "score_ext", "resultat"]].copy()
    dom_view.columns = ["match_id", "date", "equipe_id", "buts_m", "buts_e", "resultat"]
    dom_view["is_dom"] = True
    dom_view["pts"] = dom_view["resultat"].map({"1": 3, "N": 1, "2": 0}).fillna(0).astype(int)
    dom_view["victoire"] = (dom_view["resultat"] == "1").astype(int)

    ext_view = df[["match_id", "date", "equipe_ext_id", "score_ext", "score_dom", "resultat"]].copy()
    ext_view.columns = ["match_id", "date", "equipe_id", "buts_m", "buts_e", "resultat"]
    ext_view["is_dom"] = False
    ext_view["pts"] = ext_view["resultat"].map({"2": 3, "N": 1, "1": 0}).fillna(0).astype(int)
    ext_view["victoire"] = (ext_view["resultat"] == "2").astype(int)

    all_matches = pd.concat([dom_view, ext_view], ignore_index=True)
    all_matches = all_matches.sort_values(["equipe_id", "date"])

    # Rolling sur les n derniers matchs par équipe (shift pour exclure le match courant)
    grouped = all_matches.groupby("equipe_id")

    all_matches["cum_pts"] = grouped["pts"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())
    all_matches["cum_vic"] = grouped["victoire"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    all_matches["cum_bm"] = grouped["buts_m"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    all_matches["cum_be"] = grouped["buts_e"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    all_matches["match_count"] = grouped.cumcount()  # 0-based count before this match

    return all_matches


def _compute_h2h_vectorized(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Calcule les H2H vectorisés. Retourne un DataFrame indexé par match_id."""
    # Créer une clé de paire ordonnée
    df = df.copy()
    df["pair_key"] = df.apply(
        lambda r: (min(r["equipe_dom_id"], r["equipe_ext_id"]),
                   max(r["equipe_dom_id"], r["equipe_ext_id"])),
        axis=1,
    )

    # Pour chaque paire, on veut les résultats du point de vue de l'équipe dom actuelle
    # dom_won: equipe_dom_id a gagné
    df["dom_won"] = 0.0
    df["draw"] = 0.0
    df["ext_won"] = 0.0

    for idx, row in df.iterrows():
        dom_id = row["equipe_dom_id"]
        pair = row["pair_key"]
        date = row["date"]

        # Previous matches between these two teams
        mask = (df["pair_key"] == pair) & (df["date"] < date)
        prev = df.loc[mask].tail(n)

        if len(prev) == 0:
            df.at[idx, "h2h_nb"] = 0
            df.at[idx, "h2h_pct_dom"] = 0.0
            df.at[idx, "h2h_pct_nul"] = 0.0
            df.at[idx, "h2h_pct_ext"] = 0.0
            continue

        vic_dom = 0
        nuls = 0
        vic_ext = 0
        for _, pm in prev.iterrows():
            if pm["resultat"] == "N":
                nuls += 1
            elif pm["equipe_dom_id"] == dom_id:
                if pm["resultat"] == "1":
                    vic_dom += 1
                else:
                    vic_ext += 1
            else:
                if pm["resultat"] == "2":
                    vic_dom += 1
                else:
                    vic_ext += 1

        total = len(prev)
        df.at[idx, "h2h_nb"] = total
        df.at[idx, "h2h_pct_dom"] = vic_dom / total
        df.at[idx, "h2h_pct_nul"] = nuls / total
        df.at[idx, "h2h_pct_ext"] = vic_ext / total

    return df


def build_dataset(min_history: int = 10, session=None, progress: bool = True) -> pd.DataFrame:
    """Construit le dataset complet features + labels pour tous les matchs.

    Version optimisée : charge tout en mémoire et calcule les features
    avec des opérations pandas vectorisées.

    Args:
        min_history: nombre minimum de matchs par équipe avant inclusion
        session: session SQLAlchemy (optionnel, pour compatibilité)
        progress: afficher la progression

    Retourne un DataFrame avec colonnes de features + 'resultat' (label).
    """
    if progress:
        print("Chargement des matchs depuis la base...")

    df = _load_all_matches_df()
    if progress:
        print(f"  {len(df)} matchs chargés")

    # --- Forme récente (vectorisé) ---
    if progress:
        print("Calcul de la forme récente...")

    form_data = _rolling_form(df, n=5)

    # Séparer dom et ext
    form_dom = form_data[form_data["is_dom"]].set_index("match_id")
    form_ext = form_data[~form_data["is_dom"]].set_index("match_id")

    df = df.set_index("match_id")

    df["dom_forme_pts"] = form_dom["cum_pts"]
    df["dom_forme_ratio_vic"] = form_dom["cum_vic"]
    df["dom_forme_buts_m"] = form_dom["cum_bm"]
    df["dom_forme_buts_e"] = form_dom["cum_be"]
    df["dom_forme_nb"] = form_dom["match_count"].clip(upper=5)
    df["dom_match_count"] = form_dom["match_count"]

    df["ext_forme_pts"] = form_ext["cum_pts"]
    df["ext_forme_ratio_vic"] = form_ext["cum_vic"]
    df["ext_forme_buts_m"] = form_ext["cum_bm"]
    df["ext_forme_buts_e"] = form_ext["cum_be"]
    df["ext_forme_nb"] = form_ext["match_count"].clip(upper=5)
    df["ext_match_count"] = form_ext["match_count"]

    # --- Filtrer par historique minimum ---
    mask = (df["dom_match_count"] >= min_history) & (df["ext_match_count"] >= min_history)
    df = df[mask].copy()
    if progress:
        print(f"  {len(df)} matchs avec historique suffisant (>= {min_history})")

    # --- Goal stats (rolling n=10) ---
    if progress:
        print("Calcul des stats de buts...")

    form10 = _rolling_form(
        _load_all_matches_df(), n=10
    )
    form10_dom = form10[form10["is_dom"]].set_index("match_id")
    form10_ext = form10[~form10["is_dom"]].set_index("match_id")

    df["dom_moy_buts_m"] = form10_dom["cum_bm"].reindex(df.index)
    df["dom_moy_buts_e"] = form10_dom["cum_be"].reindex(df.index)
    df["ext_moy_buts_m"] = form10_ext["cum_bm"].reindex(df.index)
    df["ext_moy_buts_e"] = form10_ext["cum_be"].reindex(df.index)

    # Over 2.5 ratio (rolling 10)
    all_df = _load_all_matches_df()
    # dom view
    dom_o25 = all_df.copy()
    dom_o25["equipe_id"] = dom_o25["equipe_dom_id"]
    dom_o25["over25"] = ((dom_o25["score_dom"] + dom_o25["score_ext"]) > 2.5).astype(float)
    dom_o25 = dom_o25.sort_values(["equipe_id", "date"])
    dom_o25["cum_over25"] = dom_o25.groupby("equipe_id")["over25"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    dom_o25 = dom_o25.set_index("match_id")

    ext_o25 = all_df.copy()
    ext_o25["equipe_id"] = ext_o25["equipe_ext_id"]
    ext_o25["over25"] = ((ext_o25["score_dom"] + ext_o25["score_ext"]) > 2.5).astype(float)
    ext_o25 = ext_o25.sort_values(["equipe_id", "date"])
    ext_o25["cum_over25"] = ext_o25.groupby("equipe_id")["over25"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    ext_o25 = ext_o25.set_index("match_id")

    df["dom_over25"] = dom_o25["cum_over25"].reindex(df.index)
    df["ext_over25"] = ext_o25["cum_over25"].reindex(df.index)

    # --- Cotes features ---
    if progress:
        print("Calcul des features de cotes...")

    valid_odds = (df["cote_1"].notna()) & (df["cote_n"].notna()) & (df["cote_2"].notna())
    raw_1 = 1.0 / df["cote_1"]
    raw_n = 1.0 / df["cote_n"]
    raw_2 = 1.0 / df["cote_2"]
    total_raw = raw_1 + raw_n + raw_2

    df["prob_1"] = np.where(valid_odds, raw_1 / total_raw, np.nan)
    df["prob_n"] = np.where(valid_odds, raw_n / total_raw, np.nan)
    df["prob_2"] = np.where(valid_odds, raw_2 / total_raw, np.nan)

    min_cote = df[["cote_1", "cote_n", "cote_2"]].min(axis=1)
    df["cote_surprise"] = np.where(valid_odds, 1.0 / min_cote, np.nan)

    # --- Avantage domicile par ligue ---
    if progress:
        print("Calcul de l'avantage domicile par ligue...")

    # Calculate home advantage per league (using all historical data, approximation)
    all_for_adv = _load_all_matches_df()
    ligue_adv = (
        all_for_adv.groupby("competition_nom")["resultat"]
        .apply(lambda x: (x == "1").mean())
        .to_dict()
    )
    df["avantage_dom_ligue"] = df["competition_nom"].map(ligue_adv)

    # --- Jours de repos ---
    if progress:
        print("Calcul des jours de repos...")

    all_for_repos = _load_all_matches_df().set_index("match_id")

    # Pour chaque équipe, calculer le delta en jours depuis le match précédent
    dom_repos = all_for_repos[["date", "equipe_dom_id"]].copy()
    dom_repos.columns = ["date", "equipe_id"]
    ext_repos = all_for_repos[["date", "equipe_ext_id"]].copy()
    ext_repos.columns = ["date", "equipe_id"]

    all_repos = pd.concat([dom_repos, ext_repos])
    all_repos = all_repos.sort_values(["equipe_id", "date"])
    all_repos["prev_date"] = all_repos.groupby("equipe_id")["date"].shift(1)
    all_repos["jours_repos"] = (all_repos["date"] - all_repos["prev_date"]).dt.days

    # Map back to match_id for dom and ext
    # For dom: find the match_id row where the equipe is dom
    dom_repos_map = dom_repos.copy()
    dom_repos_map["equipe_id_match"] = dom_repos_map["equipe_id"]
    dom_repos_merged = dom_repos_map.merge(
        all_repos[["equipe_id", "date", "jours_repos"]],
        left_on=["equipe_id", "date"],
        right_on=["equipe_id", "date"],
        how="left",
    )
    # Deduplicate (take first)
    dom_repos_merged = dom_repos_merged[~dom_repos_merged.index.duplicated(keep="first")]
    df["dom_jours_repos"] = dom_repos_merged["jours_repos"].reindex(df.index)

    ext_repos_map = ext_repos.copy()
    ext_repos_merged = ext_repos_map.merge(
        all_repos[["equipe_id", "date", "jours_repos"]],
        left_on=["equipe_id", "date"],
        right_on=["equipe_id", "date"],
        how="left",
    )
    ext_repos_merged = ext_repos_merged[~ext_repos_merged.index.duplicated(keep="first")]
    df["ext_jours_repos"] = ext_repos_merged["jours_repos"].reindex(df.index)

    # --- H2H (simplified: skip for batch mode, too expensive row-by-row) ---
    # We set h2h to 0/NaN — can be computed later with build_match_features for smaller subsets
    if progress:
        print("H2H: utilisation de valeurs par défaut (calcul complet disponible via build_match_features)...")

    df["h2h_nb"] = 0
    df["h2h_pct_dom"] = 0.0
    df["h2h_pct_nul"] = 0.0
    df["h2h_pct_ext"] = 0.0

    # --- Classement (simplifié: différence de points) ---
    # Full dynamic standings are too expensive for 81k matches
    # We use the cumulative points in the season as proxy
    df["dom_classement_pts"] = df["dom_forme_pts"]  # Proxy
    df["ext_classement_pts"] = df["ext_forme_pts"]
    df["dom_position"] = np.nan
    df["ext_position"] = np.nan
    df["diff_position"] = np.nan

    # --- Sélection des colonnes finales ---
    feature_cols = [
        "dom_forme_pts", "dom_forme_ratio_vic", "dom_forme_buts_m", "dom_forme_buts_e", "dom_forme_nb",
        "ext_forme_pts", "ext_forme_ratio_vic", "ext_forme_buts_m", "ext_forme_buts_e", "ext_forme_nb",
        "h2h_nb", "h2h_pct_dom", "h2h_pct_nul", "h2h_pct_ext",
        "dom_position", "ext_position", "diff_position",
        "dom_classement_pts", "ext_classement_pts",
        "dom_moy_buts_m", "dom_moy_buts_e", "dom_over25",
        "ext_moy_buts_m", "ext_moy_buts_e", "ext_over25",
        "prob_1", "prob_n", "prob_2", "cote_surprise",
        "cote_1", "cote_n", "cote_2",
        "dom_jours_repos", "ext_jours_repos",
        "avantage_dom_ligue",
    ]
    meta_cols = ["date", "equipe_dom_id", "equipe_ext_id", "competition_id", "resultat"]

    # Reset index to get match_id back as column
    df = df.reset_index()
    all_cols = ["match_id"] + meta_cols + feature_cols
    existing = [c for c in all_cols if c in df.columns]
    result = df[existing].copy()

    if progress:
        print(f"Dataset final: {result.shape}")

    return result


def build_dataset_precise(match_ids: list = None, session=None) -> pd.DataFrame:
    """Construit un dataset précis pour un sous-ensemble de matchs.

    Utilise build_match_features (requêtes individuelles) pour un calcul
    exact de toutes les features y compris H2H et classement.
    Plus lent mais plus précis — à utiliser pour de petits ensembles.

    Args:
        match_ids: liste d'IDs de matchs (si None, prend les 1000 derniers)
        session: session SQLAlchemy
    """
    close = session is None
    if close:
        session = SessionLocal()

    try:
        if match_ids is None:
            stmt = (
                select(Match)
                .where(Match.resultat.isnot(None))
                .order_by(Match.date.desc())
                .limit(1000)
            )
        else:
            stmt = select(Match).where(Match.id.in_(match_ids))

        matchs = session.execute(stmt).scalars().all()

        records = []
        for i, match in enumerate(matchs):
            try:
                features = build_match_features(match, session=session)
                features["match_id"] = match.id
                features["date"] = match.date
                features["equipe_dom_id"] = match.equipe_dom_id
                features["equipe_ext_id"] = match.equipe_ext_id
                features["competition_id"] = match.competition_id
                features["resultat"] = match.resultat
                records.append(features)
            except Exception:
                pass

        return pd.DataFrame(records)
    finally:
        if close:
            session.close()


def split_by_date(df: pd.DataFrame, train_end: str, val_end: str):
    """Split le dataset en train/validation/test par date (walk-forward).

    Args:
        df: DataFrame avec colonne 'date'
        train_end: date de fin du train (format 'YYYY-MM-DD')
        val_end: date de fin de la validation (format 'YYYY-MM-DD')

    Returns:
        tuple (train_df, val_df, test_df)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    train_df = df[df["date"] <= train_end_dt].copy()
    val_df = df[(df["date"] > train_end_dt) & (df["date"] <= val_end_dt)].copy()
    test_df = df[df["date"] > val_end_dt].copy()

    return train_df, val_df, test_df


def save_dataset(df: pd.DataFrame, path: str = "data/dataset.csv") -> None:
    """Sauvegarde le dataset en CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset(path: str = "data/dataset.csv") -> pd.DataFrame:
    """Charge un dataset depuis un CSV."""
    return pd.read_csv(path, parse_dates=["date"])
