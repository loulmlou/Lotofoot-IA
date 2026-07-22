"""Calcul de features par match pour le moteur de prédiction."""

import math
from datetime import datetime, timedelta

from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import Session

from database.models import Match, Cote, Competition, Equipe
from database.connection import SessionLocal


def compute_form(equipe_id: int, date, n: int = 5, session: Session = None) -> dict:
    """Calcule la forme récente d'une équipe sur ses n derniers matchs.

    Retourne points, ratio victoires, buts marqués/encaissés,
    forme domicile et forme extérieur séparément.
    """
    close = session is None
    if close:
        session = SessionLocal()

    try:
        # Tous les matchs avant la date
        stmt = (
            select(Match)
            .where(
                and_(
                    or_(
                        Match.equipe_dom_id == equipe_id,
                        Match.equipe_ext_id == equipe_id,
                    ),
                    Match.date < date,
                    Match.resultat.isnot(None),
                )
            )
            .order_by(Match.date.desc())
            .limit(n)
        )
        matchs = session.execute(stmt).scalars().all()

        if not matchs:
            return {
                "points": 0, "ratio_victoires": 0.0,
                "buts_marques": 0.0, "buts_encaisses": 0.0,
                "nb_matchs": 0,
                "forme_dom": {"points": 0, "nb": 0},
                "forme_ext": {"points": 0, "nb": 0},
            }

        points = 0
        victoires = 0
        buts_m = 0
        buts_e = 0
        dom_pts = 0
        dom_nb = 0
        ext_pts = 0
        ext_nb = 0

        for m in matchs:
            is_dom = m.equipe_dom_id == equipe_id
            if is_dom:
                bm = m.score_dom or 0
                be = m.score_ext or 0
            else:
                bm = m.score_ext or 0
                be = m.score_dom or 0

            buts_m += bm
            buts_e += be

            if is_dom:
                if m.resultat == "1":
                    pts = 3
                    victoires += 1
                elif m.resultat == "N":
                    pts = 1
                else:
                    pts = 0
                dom_pts += pts
                dom_nb += 1
            else:
                if m.resultat == "2":
                    pts = 3
                    victoires += 1
                elif m.resultat == "N":
                    pts = 1
                else:
                    pts = 0
                ext_pts += pts
                ext_nb += 1

            points += pts

        total = len(matchs)
        return {
            "points": points,
            "ratio_victoires": victoires / total,
            "buts_marques": buts_m / total,
            "buts_encaisses": buts_e / total,
            "nb_matchs": total,
            "forme_dom": {"points": dom_pts, "nb": dom_nb},
            "forme_ext": {"points": ext_pts, "nb": ext_nb},
        }
    finally:
        if close:
            session.close()


def compute_h2h(equipe_dom_id: int, equipe_ext_id: int, date, n: int = 5,
                session: Session = None) -> dict:
    """Calcule les confrontations directes entre 2 équipes.

    Retourne % victoire dom, % nul, % victoire ext sur les n dernières rencontres.
    """
    close = session is None
    if close:
        session = SessionLocal()

    try:
        stmt = (
            select(Match)
            .where(
                and_(
                    or_(
                        and_(
                            Match.equipe_dom_id == equipe_dom_id,
                            Match.equipe_ext_id == equipe_ext_id,
                        ),
                        and_(
                            Match.equipe_dom_id == equipe_ext_id,
                            Match.equipe_ext_id == equipe_dom_id,
                        ),
                    ),
                    Match.date < date,
                    Match.resultat.isnot(None),
                )
            )
            .order_by(Match.date.desc())
            .limit(n)
        )
        matchs = session.execute(stmt).scalars().all()

        if not matchs:
            return {
                "nb_matchs": 0,
                "pct_dom": 0.0,
                "pct_nul": 0.0,
                "pct_ext": 0.0,
            }

        # Victoires de equipe_dom_id (quelle que soit la configuration dom/ext)
        vic_dom = 0
        nuls = 0
        vic_ext = 0

        for m in matchs:
            if m.resultat == "N":
                nuls += 1
            elif m.equipe_dom_id == equipe_dom_id:
                if m.resultat == "1":
                    vic_dom += 1
                else:
                    vic_ext += 1
            else:
                # Configuration inversée
                if m.resultat == "2":
                    vic_dom += 1
                else:
                    vic_ext += 1

        total = len(matchs)
        return {
            "nb_matchs": total,
            "pct_dom": vic_dom / total,
            "pct_nul": nuls / total,
            "pct_ext": vic_ext / total,
        }
    finally:
        if close:
            session.close()


def compute_standing(equipe_id: int, competition_id: int, date,
                     session: Session = None) -> dict:
    """Calcule le classement simulé d'une équipe au moment du match.

    Parcourt tous les matchs de la compétition avant la date pour reconstruire
    le classement.
    """
    close = session is None
    if close:
        session = SessionLocal()

    try:
        # Déterminer la saison en cours à cette date
        stmt = (
            select(Match.saison)
            .where(
                and_(
                    Match.competition_id == competition_id,
                    Match.date <= date,
                    Match.saison.isnot(None),
                )
            )
            .order_by(Match.date.desc())
            .limit(1)
        )
        result = session.execute(stmt).scalar()
        if not result:
            return {"position": None, "points": 0, "matchs_joues": 0}

        saison = result

        # Tous les matchs de cette compétition/saison avant la date
        stmt = (
            select(Match)
            .where(
                and_(
                    Match.competition_id == competition_id,
                    Match.saison == saison,
                    Match.date < date,
                    Match.resultat.isnot(None),
                )
            )
        )
        matchs = session.execute(stmt).scalars().all()

        if not matchs:
            return {"position": None, "points": 0, "matchs_joues": 0}

        # Calculer les points de chaque équipe
        standings = {}  # equipe_id -> {points, diff, matchs}
        for m in matchs:
            for eid in [m.equipe_dom_id, m.equipe_ext_id]:
                if eid not in standings:
                    standings[eid] = {"points": 0, "diff": 0, "matchs": 0}

            sd = m.score_dom or 0
            se = m.score_ext or 0

            if m.resultat == "1":
                standings[m.equipe_dom_id]["points"] += 3
            elif m.resultat == "N":
                standings[m.equipe_dom_id]["points"] += 1
                standings[m.equipe_ext_id]["points"] += 1
            else:
                standings[m.equipe_ext_id]["points"] += 3

            standings[m.equipe_dom_id]["diff"] += sd - se
            standings[m.equipe_ext_id]["diff"] += se - sd
            standings[m.equipe_dom_id]["matchs"] += 1
            standings[m.equipe_ext_id]["matchs"] += 1

        # Trier par points puis différence de buts
        classement = sorted(
            standings.items(),
            key=lambda x: (x[1]["points"], x[1]["diff"]),
            reverse=True,
        )

        position = None
        pts = 0
        mj = 0
        for i, (eid, data) in enumerate(classement, 1):
            if eid == equipe_id:
                position = i
                pts = data["points"]
                mj = data["matchs"]
                break

        return {
            "position": position,
            "points": pts,
            "matchs_joues": mj,
            "nb_equipes": len(classement),
        }
    finally:
        if close:
            session.close()


def compute_goal_stats(equipe_id: int, date, n: int = 10,
                       session: Session = None) -> dict:
    """Calcule les statistiques de buts d'une équipe sur les n derniers matchs.

    Retourne moyenne buts marqués/encaissés (dom et ext), ratio over/under 2.5.
    """
    close = session is None
    if close:
        session = SessionLocal()

    try:
        stmt = (
            select(Match)
            .where(
                and_(
                    or_(
                        Match.equipe_dom_id == equipe_id,
                        Match.equipe_ext_id == equipe_id,
                    ),
                    Match.date < date,
                    Match.score_dom.isnot(None),
                    Match.score_ext.isnot(None),
                )
            )
            .order_by(Match.date.desc())
            .limit(n)
        )
        matchs = session.execute(stmt).scalars().all()

        if not matchs:
            return {
                "moy_buts_marques": 0.0,
                "moy_buts_encaisses": 0.0,
                "moy_buts_marques_dom": 0.0,
                "moy_buts_encaisses_dom": 0.0,
                "moy_buts_marques_ext": 0.0,
                "moy_buts_encaisses_ext": 0.0,
                "over_2_5_ratio": 0.0,
                "nb_matchs": 0,
            }

        bm_total, be_total = 0, 0
        bm_dom, be_dom, nb_dom = 0, 0, 0
        bm_ext, be_ext, nb_ext = 0, 0, 0
        over = 0

        for m in matchs:
            sd = m.score_dom or 0
            se = m.score_ext or 0
            total_buts = sd + se

            if total_buts > 2.5:
                over += 1

            if m.equipe_dom_id == equipe_id:
                bm_total += sd
                be_total += se
                bm_dom += sd
                be_dom += se
                nb_dom += 1
            else:
                bm_total += se
                be_total += sd
                bm_ext += se
                be_ext += sd
                nb_ext += 1

        total = len(matchs)
        return {
            "moy_buts_marques": bm_total / total,
            "moy_buts_encaisses": be_total / total,
            "moy_buts_marques_dom": bm_dom / nb_dom if nb_dom else 0.0,
            "moy_buts_encaisses_dom": be_dom / nb_dom if nb_dom else 0.0,
            "moy_buts_marques_ext": bm_ext / nb_ext if nb_ext else 0.0,
            "moy_buts_encaisses_ext": be_ext / nb_ext if nb_ext else 0.0,
            "over_2_5_ratio": over / total,
            "nb_matchs": total,
        }
    finally:
        if close:
            session.close()


def compute_odds_features(cote) -> dict:
    """Calcule les features dérivées des cotes.

    cote: objet Cote ou dict avec cote_1, cote_n, cote_2.
    Retourne probabilités implicites normalisées et cote surprise.
    """
    if cote is None:
        return {
            "prob_1": None, "prob_n": None, "prob_2": None,
            "cote_surprise": None,
            "cote_1": None, "cote_n": None, "cote_2": None,
        }

    if isinstance(cote, dict):
        c1 = cote.get("cote_1")
        cn = cote.get("cote_n")
        c2 = cote.get("cote_2")
    else:
        c1 = cote.cote_1
        cn = cote.cote_n
        c2 = cote.cote_2

    if not c1 or not cn or not c2:
        return {
            "prob_1": None, "prob_n": None, "prob_2": None,
            "cote_surprise": None,
            "cote_1": c1, "cote_n": cn, "cote_2": c2,
        }

    # Probabilités implicites brutes
    raw_1 = 1.0 / c1
    raw_n = 1.0 / cn
    raw_2 = 1.0 / c2
    total = raw_1 + raw_n + raw_2

    # Normalisation (supprime la marge du bookmaker)
    prob_1 = raw_1 / total
    prob_n = raw_n / total
    prob_2 = raw_2 / total

    # Cote surprise = inverse du favori (plus c'est grand, plus le favori est fort)
    cote_favorite = min(c1, cn, c2)
    cote_surprise = 1.0 / cote_favorite

    return {
        "prob_1": prob_1,
        "prob_n": prob_n,
        "prob_2": prob_2,
        "cote_surprise": cote_surprise,
        "cote_1": c1,
        "cote_n": cn,
        "cote_2": c2,
    }


def compute_context(equipe_id: int, date, competition_id: int = None,
                    session: Session = None) -> dict:
    """Calcule les features de contexte : repos et avantage domicile par ligue."""
    close = session is None
    if close:
        session = SessionLocal()

    try:
        # Jours depuis le dernier match
        stmt = (
            select(Match.date)
            .where(
                and_(
                    or_(
                        Match.equipe_dom_id == equipe_id,
                        Match.equipe_ext_id == equipe_id,
                    ),
                    Match.date < date,
                )
            )
            .order_by(Match.date.desc())
            .limit(1)
        )
        last_date = session.execute(stmt).scalar()

        if last_date:
            if isinstance(date, datetime):
                d = date
            else:
                d = datetime.combine(date, datetime.min.time())
            if isinstance(last_date, datetime):
                ld = last_date
            else:
                ld = datetime.combine(last_date, datetime.min.time())
            jours_repos = (d - ld).days
        else:
            jours_repos = None

        # Avantage domicile historique par ligue
        avantage_dom = None
        if competition_id:
            # Récupérer le nom de la compétition (sans la saison)
            comp = session.get(Competition, competition_id)
            if comp:
                # Tous les matchs de cette ligue (toutes saisons)
                stmt_comps = (
                    select(Competition.id)
                    .where(Competition.nom == comp.nom)
                )
                comp_ids = [r for r in session.execute(stmt_comps).scalars().all()]

                stmt_total = (
                    select(func.count())
                    .select_from(Match)
                    .where(
                        and_(
                            Match.competition_id.in_(comp_ids),
                            Match.date < date,
                            Match.resultat.isnot(None),
                        )
                    )
                )
                total = session.execute(stmt_total).scalar() or 0

                stmt_dom = (
                    select(func.count())
                    .select_from(Match)
                    .where(
                        and_(
                            Match.competition_id.in_(comp_ids),
                            Match.date < date,
                            Match.resultat == "1",
                        )
                    )
                )
                dom_wins = session.execute(stmt_dom).scalar() or 0

                avantage_dom = dom_wins / total if total > 0 else None

        return {
            "jours_repos": jours_repos,
            "avantage_dom_ligue": avantage_dom,
        }
    finally:
        if close:
            session.close()


def build_match_features(match, session: Session = None) -> dict:
    """Agrège toutes les features pour un match donné.

    Retourne un dict plat avec toutes les features numériques.
    """
    close = session is None
    if close:
        session = SessionLocal()

    try:
        date = match.date
        dom_id = match.equipe_dom_id
        ext_id = match.equipe_ext_id
        comp_id = match.competition_id

        # Forme récente
        forme_dom = compute_form(dom_id, date, n=5, session=session)
        forme_ext = compute_form(ext_id, date, n=5, session=session)

        # Confrontations directes
        h2h = compute_h2h(dom_id, ext_id, date, n=5, session=session)

        # Classement
        standing_dom = compute_standing(dom_id, comp_id, date, session=session) if comp_id else {}
        standing_ext = compute_standing(ext_id, comp_id, date, session=session) if comp_id else {}

        # Stats de buts
        goals_dom = compute_goal_stats(dom_id, date, n=10, session=session)
        goals_ext = compute_goal_stats(ext_id, date, n=10, session=session)

        # Cotes
        odds = compute_odds_features(match.cotes)

        # Contexte
        ctx_dom = compute_context(dom_id, date, comp_id, session=session)
        ctx_ext = compute_context(ext_id, date, comp_id, session=session)

        # Aplatir en dict de features
        features = {
            # Forme domicile
            "dom_forme_pts": forme_dom["points"],
            "dom_forme_ratio_vic": forme_dom["ratio_victoires"],
            "dom_forme_buts_m": forme_dom["buts_marques"],
            "dom_forme_buts_e": forme_dom["buts_encaisses"],
            "dom_forme_nb": forme_dom["nb_matchs"],

            # Forme extérieur
            "ext_forme_pts": forme_ext["points"],
            "ext_forme_ratio_vic": forme_ext["ratio_victoires"],
            "ext_forme_buts_m": forme_ext["buts_marques"],
            "ext_forme_buts_e": forme_ext["buts_encaisses"],
            "ext_forme_nb": forme_ext["nb_matchs"],

            # H2H
            "h2h_nb": h2h["nb_matchs"],
            "h2h_pct_dom": h2h["pct_dom"],
            "h2h_pct_nul": h2h["pct_nul"],
            "h2h_pct_ext": h2h["pct_ext"],

            # Classement
            "dom_position": standing_dom.get("position"),
            "ext_position": standing_ext.get("position"),
            "diff_position": (
                (standing_ext.get("position") or 0) - (standing_dom.get("position") or 0)
            ) if standing_dom.get("position") and standing_ext.get("position") else None,
            "dom_classement_pts": standing_dom.get("points", 0),
            "ext_classement_pts": standing_ext.get("points", 0),

            # Buts
            "dom_moy_buts_m": goals_dom["moy_buts_marques"],
            "dom_moy_buts_e": goals_dom["moy_buts_encaisses"],
            "dom_over25": goals_dom["over_2_5_ratio"],
            "ext_moy_buts_m": goals_ext["moy_buts_marques"],
            "ext_moy_buts_e": goals_ext["moy_buts_encaisses"],
            "ext_over25": goals_ext["over_2_5_ratio"],

            # Cotes
            "prob_1": odds["prob_1"],
            "prob_n": odds["prob_n"],
            "prob_2": odds["prob_2"],
            "cote_surprise": odds["cote_surprise"],
            "cote_1": odds["cote_1"],
            "cote_n": odds["cote_n"],
            "cote_2": odds["cote_2"],

            # Contexte
            "dom_jours_repos": ctx_dom["jours_repos"],
            "ext_jours_repos": ctx_ext["jours_repos"],
            "avantage_dom_ligue": ctx_dom["avantage_dom_ligue"],
        }

        return features
    finally:
        if close:
            session.close()
