"""Analyse exploratoire des matchs — statistiques descriptives."""

import pandas as pd
from sqlalchemy import select, func, and_

from database.models import Match, Cote, Competition
from database.connection import SessionLocal


def home_advantage_by_league() -> pd.DataFrame:
    """Calcule l'avantage domicile par ligue.

    Retourne un DataFrame avec colonnes :
    ligue, pays, nb_matchs, pct_dom, pct_nul, pct_ext.
    """
    session = SessionLocal()
    try:
        stmt = (
            select(
                Competition.nom,
                Competition.pays,
                Match.resultat,
                func.count().label("nb"),
            )
            .join(Competition, Match.competition_id == Competition.id)
            .where(Match.resultat.isnot(None))
            .group_by(Competition.nom, Competition.pays, Match.resultat)
        )
        rows = session.execute(stmt).all()

        data = {}
        for nom, pays, resultat, nb in rows:
            key = (nom, pays)
            if key not in data:
                data[key] = {"1": 0, "N": 0, "2": 0}
            if resultat in data[key]:
                data[key][resultat] += nb

        records = []
        for (nom, pays), counts in data.items():
            total = counts["1"] + counts["N"] + counts["2"]
            if total > 0:
                records.append({
                    "ligue": nom,
                    "pays": pays,
                    "nb_matchs": total,
                    "pct_dom": counts["1"] / total,
                    "pct_nul": counts["N"] / total,
                    "pct_ext": counts["2"] / total,
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("pct_dom", ascending=False).reset_index(drop=True)
        return df
    finally:
        session.close()


def odds_calibration() -> pd.DataFrame:
    """Analyse la calibration des bookmakers : cotes vs résultats réels.

    Regroupe les matchs par tranche de probabilité implicite et compare
    avec le taux de réalisation réel.

    Retourne un DataFrame avec colonnes :
    tranche_prob, nb_matchs, prob_implicite_moy, taux_realisation.
    """
    session = SessionLocal()
    try:
        stmt = (
            select(Match, Cote)
            .join(Cote, Cote.match_id == Match.id)
            .where(
                and_(
                    Match.resultat.isnot(None),
                    Cote.cote_1.isnot(None),
                    Cote.cote_n.isnot(None),
                    Cote.cote_2.isnot(None),
                )
            )
        )
        rows = session.execute(stmt).all()

        # Pour chaque résultat possible, calculer prob implicite vs réalisation
        records = []
        for match, cote in rows:
            raw_1 = 1.0 / cote.cote_1
            raw_n = 1.0 / cote.cote_n
            raw_2 = 1.0 / cote.cote_2
            total = raw_1 + raw_n + raw_2

            # On analyse la probabilité du favori (plus petite cote)
            prob_1 = raw_1 / total
            prob_n = raw_n / total
            prob_2 = raw_2 / total

            # Pour chaque issue
            for prob, res_attendu in [(prob_1, "1"), (prob_n, "N"), (prob_2, "2")]:
                records.append({
                    "prob_implicite": prob,
                    "resultat_attendu": res_attendu,
                    "realise": 1 if match.resultat == res_attendu else 0,
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Regrouper par tranches de 10%
        df["tranche"] = pd.cut(
            df["prob_implicite"],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"],
        )

        result = (
            df.groupby("tranche", observed=False)
            .agg(
                nb_matchs=("realise", "count"),
                prob_implicite_moy=("prob_implicite", "mean"),
                taux_realisation=("realise", "mean"),
            )
            .reset_index()
            .rename(columns={"tranche": "tranche_prob"})
        )

        return result
    finally:
        session.close()


def results_distribution(competition_id: int = None, saison: str = None) -> dict:
    """Distribution des résultats, optionnellement filtrée par compétition/saison.

    Retourne {total, count_1, count_n, count_2, pct_1, pct_n, pct_2}.
    """
    session = SessionLocal()
    try:
        conditions = [Match.resultat.isnot(None)]
        if competition_id:
            conditions.append(Match.competition_id == competition_id)
        if saison:
            conditions.append(Match.saison == saison)

        stmt = (
            select(Match.resultat, func.count().label("nb"))
            .where(and_(*conditions))
            .group_by(Match.resultat)
        )
        rows = session.execute(stmt).all()

        counts = {"1": 0, "N": 0, "2": 0}
        for resultat, nb in rows:
            if resultat in counts:
                counts[resultat] = nb

        total = sum(counts.values())
        return {
            "total": total,
            "count_1": counts["1"],
            "count_n": counts["N"],
            "count_2": counts["2"],
            "pct_1": counts["1"] / total if total else 0.0,
            "pct_n": counts["N"] / total if total else 0.0,
            "pct_2": counts["2"] / total if total else 0.0,
        }
    finally:
        session.close()


def feature_correlation_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la matrice de corrélation des features numériques.

    Prend un DataFrame de features (colonnes numériques) et retourne
    la matrice de corrélation de Pearson.
    """
    numeric_cols = features_df.select_dtypes(include=["number"]).columns
    return features_df[numeric_cols].corr()
