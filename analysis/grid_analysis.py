"""Analyse statistique des grilles Loto Foot historiques."""

import math
from collections import Counter

from sqlalchemy import select

from database.models import GrilleLotoFoot, MatchGrille, StatistiqueGrille
from database.connection import SessionLocal


def _compute_entropy(resultats: str) -> float:
    """Calcule l'entropie de Shannon d'une séquence de résultats."""
    if not resultats:
        return 0.0

    n = len(resultats)
    counts = Counter(resultats)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_alternance(resultats: str) -> float:
    """Calcule le taux d'alternance (changements de résultat consécutifs)."""
    if not resultats or len(resultats) < 2:
        return 0.0

    changements = sum(
        1 for i in range(1, len(resultats)) if resultats[i] != resultats[i - 1]
    )
    return changements / (len(resultats) - 1)


def _compute_plus_longue_suite(resultats: str) -> int:
    """Calcule la plus longue suite du même résultat consécutif."""
    if not resultats:
        return 0

    max_suite = 1
    suite = 1
    for i in range(1, len(resultats)):
        if resultats[i] == resultats[i - 1]:
            suite += 1
            max_suite = max(max_suite, suite)
        else:
            suite = 1
    return max_suite


def _compute_indice_chaos(resultats: str, cotes_matchs: list = None) -> float:
    """Calcule l'indice de chaos : proportion de résultats surprenants.

    Un résultat est surprenant si la cote du résultat réel > 3.0.
    Sans cotes, on utilise une heuristique basée sur la distribution.
    """
    if not resultats:
        return 0.0

    if cotes_matchs:
        surprises = 0
        for i, res in enumerate(resultats):
            if i < len(cotes_matchs) and cotes_matchs[i]:
                cote = cotes_matchs[i]
                if res == "1" and cote.get("cote_1", 0) and cote["cote_1"] > 3.0:
                    surprises += 1
                elif res == "N" and cote.get("cote_n", 0) and cote["cote_n"] > 3.0:
                    surprises += 1
                elif res == "2" and cote.get("cote_2", 0) and cote["cote_2"] > 3.0:
                    surprises += 1
        return surprises / len(resultats)

    # Heuristique sans cotes : les nuls et victoires extérieures sont
    # considérés comme plus "chaotiques"
    surprises = sum(1 for r in resultats if r in ("N", "2"))
    return surprises / len(resultats)


def compute_grid_stats(grille, session=None) -> StatistiqueGrille:
    """Calcule toutes les statistiques pour une grille Loto Foot.

    Retourne un objet StatistiqueGrille prêt à être persisté.
    """
    close = session is None
    if close:
        session = SessionLocal()

    try:
        resultats = grille.resultats or ""

        # Comptage 1/N/2
        nombre_1 = resultats.count("1")
        nombre_n = resultats.count("N")
        nombre_2 = resultats.count("2")

        # Profil (ex: "5-1-1")
        profil = f"{nombre_1}-{nombre_n}-{nombre_2}"

        # Entropie
        entropie = _compute_entropy(resultats)

        # Alternance
        alternance = _compute_alternance(resultats)

        # Plus longue suite
        plus_longue_suite = _compute_plus_longue_suite(resultats)

        # Indice de chaos (avec cotes si disponibles)
        cotes_matchs = []
        stmt = (
            select(MatchGrille)
            .where(MatchGrille.grille_id == grille.id)
            .order_by(MatchGrille.position)
        )
        matchs_grille = session.execute(stmt).scalars().all()

        for mg in matchs_grille:
            if mg.match and mg.match.cotes:
                cote = mg.match.cotes
                cotes_matchs.append({
                    "cote_1": cote.cote_1,
                    "cote_n": cote.cote_n,
                    "cote_2": cote.cote_2,
                })
            else:
                cotes_matchs.append(None)

        indice_chaos = _compute_indice_chaos(resultats, cotes_matchs if cotes_matchs else None)

        stat = StatistiqueGrille(
            grille_id=grille.id,
            nombre_1=nombre_1,
            nombre_n=nombre_n,
            nombre_2=nombre_2,
            indice_chaos=indice_chaos,
            entropie=entropie,
            alternance=alternance,
            profil=profil,
            plus_longue_suite=plus_longue_suite,
        )
        return stat
    finally:
        if close:
            session.close()


def compute_all_grid_stats() -> None:
    """Calcule et persiste les statistiques pour toutes les grilles en base."""
    session = SessionLocal()
    try:
        stmt = select(GrilleLotoFoot)
        grilles = session.execute(stmt).scalars().all()

        for grille in grilles:
            # Vérifier si stat existe déjà
            existing = session.execute(
                select(StatistiqueGrille)
                .where(StatistiqueGrille.grille_id == grille.id)
            ).scalar()

            if existing:
                # Mettre à jour
                stat = compute_grid_stats(grille, session=session)
                existing.nombre_1 = stat.nombre_1
                existing.nombre_n = stat.nombre_n
                existing.nombre_2 = stat.nombre_2
                existing.indice_chaos = stat.indice_chaos
                existing.entropie = stat.entropie
                existing.alternance = stat.alternance
                existing.profil = stat.profil
                existing.plus_longue_suite = stat.plus_longue_suite
            else:
                stat = compute_grid_stats(grille, session=session)
                session.add(stat)

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_distribution_by_type() -> dict:
    """Retourne les statistiques agrégées par type de grille.

    Retourne un dict {type_grille: {nb_grilles, moy_1, moy_n, moy_2,
    moy_entropie, moy_chaos, moy_alternance}}.
    """
    session = SessionLocal()
    try:
        stmt = (
            select(GrilleLotoFoot, StatistiqueGrille)
            .outerjoin(
                StatistiqueGrille,
                StatistiqueGrille.grille_id == GrilleLotoFoot.id,
            )
        )
        rows = session.execute(stmt).all()

        by_type = {}
        for grille, stat in rows:
            t = grille.type_grille
            if t not in by_type:
                by_type[t] = {
                    "nb_grilles": 0,
                    "total_1": 0, "total_n": 0, "total_2": 0,
                    "total_entropie": 0.0, "total_chaos": 0.0,
                    "total_alternance": 0.0,
                }

            by_type[t]["nb_grilles"] += 1
            if stat:
                by_type[t]["total_1"] += stat.nombre_1 or 0
                by_type[t]["total_n"] += stat.nombre_n or 0
                by_type[t]["total_2"] += stat.nombre_2 or 0
                by_type[t]["total_entropie"] += stat.entropie or 0.0
                by_type[t]["total_chaos"] += stat.indice_chaos or 0.0
                by_type[t]["total_alternance"] += stat.alternance or 0.0

        result = {}
        for t, data in by_type.items():
            n = data["nb_grilles"]
            if n > 0:
                result[t] = {
                    "nb_grilles": n,
                    "moy_1": data["total_1"] / n,
                    "moy_n": data["total_n"] / n,
                    "moy_2": data["total_2"] / n,
                    "moy_entropie": data["total_entropie"] / n,
                    "moy_chaos": data["total_chaos"] / n,
                    "moy_alternance": data["total_alternance"] / n,
                }

        return result
    finally:
        session.close()
