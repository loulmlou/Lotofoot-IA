"""Import des CSV football-data.co.uk en base de données SQLAlchemy."""

import os
import glob
import pandas as pd
from datetime import datetime
from loguru import logger
from sqlalchemy import select

from config.settings import FOOTBALL_DATA_LEAGUES, RAW_FOOTBALL_DIR
from database.connection import init_db, SessionLocal
from database.models import Competition, Equipe, Match, Cote


# Colonnes de cotes disponibles dans les CSV (bookmakers)
ODDS_COLUMNS_HOME = ["B365H", "BWH", "IWH", "PSH", "WHH", "VCH"]
ODDS_COLUMNS_DRAW = ["B365D", "BWD", "IWD", "PSD", "WHD", "VCD"]
ODDS_COLUMNS_AWAY = ["B365A", "BWA", "IWA", "PSA", "WHA", "VCA"]

# Formats de date courants dans les CSV
DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%y"]


def parse_date(date_str: str) -> datetime | None:
    """Parse une date depuis les différents formats utilisés dans les CSV."""
    if pd.isna(date_str):
        return None
    date_str = str(date_str).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    logger.warning(f"Format de date inconnu : {date_str}")
    return None


def mean_odds(row: pd.Series, columns: list[str]) -> float | None:
    """Calcule la moyenne des cotes disponibles pour un bookmaker."""
    values = []
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            try:
                val = float(row[col])
                if val > 1.0:
                    values.append(val)
            except (ValueError, TypeError):
                continue
    return round(sum(values) / len(values), 2) if values else None


def extract_season_from_filename(filename: str) -> str:
    """Extrait le code saison depuis le nom de fichier (ex: E0_2324.csv → 2324)."""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    return parts[-1] if len(parts) >= 2 else ""


def extract_league_from_filename(filename: str) -> str:
    """Extrait le code ligue depuis le nom de fichier (ex: E0_2324.csv → E0)."""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    return parts[0] if parts else ""


def season_label(code: str) -> str:
    """Convertit un code saison en label (ex: 2324 → 2023/2024)."""
    if len(code) != 4:
        return code
    y1 = int(code[:2])
    y2 = int(code[2:])
    century1 = 2000 if y1 < 90 else 1900
    century2 = 2000 if y2 < 90 else 1900
    return f"{century1 + y1}/{century2 + y2}"


def ftr_to_resultat(ftr: str) -> str | None:
    """Convertit le code FTR (H/D/A) en résultat LotoFoot (1/N/2)."""
    mapping = {"H": "1", "D": "N", "A": "2"}
    return mapping.get(str(ftr).strip().upper())


def get_or_create_competition(session, league_code: str, saison: str) -> Competition:
    """Récupère ou crée une compétition."""
    league_info = FOOTBALL_DATA_LEAGUES.get(league_code, {})
    nom = league_info.get("nom", league_code)
    pays = league_info.get("pays", "")

    comp = session.execute(
        select(Competition).where(
            Competition.nom == nom,
            Competition.saison == saison,
        )
    ).scalar_one_or_none()

    if comp is None:
        comp = Competition(nom=nom, pays=pays, saison=saison)
        session.add(comp)
        session.flush()

    return comp


def get_or_create_equipe(session, nom: str, pays: str) -> Equipe:
    """Récupère ou crée une équipe."""
    equipe = session.execute(
        select(Equipe).where(Equipe.nom == nom)
    ).scalar_one_or_none()

    if equipe is None:
        equipe = Equipe(nom=nom, pays=pays)
        session.add(equipe)
        session.flush()

    return equipe


def match_exists(session, date: datetime, equipe_dom_id: int, equipe_ext_id: int) -> bool:
    """Vérifie si un match existe déjà en base."""
    result = session.execute(
        select(Match).where(
            Match.date == date,
            Match.equipe_dom_id == equipe_dom_id,
            Match.equipe_ext_id == equipe_ext_id,
        )
    ).scalar_one_or_none()
    return result is not None


def import_csv(filepath: str, session=None) -> dict:
    """Importe un fichier CSV en base de données.

    Returns:
        Dict avec les compteurs : matchs_inserted, matchs_skipped, errors.
    """
    stats = {"matchs_inserted": 0, "matchs_skipped": 0, "errors": 0}

    league_code = extract_league_from_filename(filepath)
    season_code = extract_season_from_filename(filepath)
    saison = season_label(season_code)
    league_info = FOOTBALL_DATA_LEAGUES.get(league_code, {})
    pays = league_info.get("pays", "")

    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:
            logger.error(f"Impossible de lire {filepath} : {e}")
            stats["errors"] += 1
            return stats

    # Vérifier que les colonnes essentielles existent
    required = ["Date", "HomeTeam", "AwayTeam", "FTR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Colonnes manquantes dans {filepath} : {missing}")
        stats["errors"] += 1
        return stats

    own_session = session is None
    if own_session:
        session = SessionLocal()

    try:
        competition = get_or_create_competition(session, league_code, saison)

        for _, row in df.iterrows():
            date = parse_date(row.get("Date"))
            home = str(row.get("HomeTeam", "")).strip()
            away = str(row.get("AwayTeam", "")).strip()
            ftr = row.get("FTR")

            if date is None or not home or not away or pd.isna(ftr):
                stats["errors"] += 1
                continue

            equipe_dom = get_or_create_equipe(session, home, pays)
            equipe_ext = get_or_create_equipe(session, away, pays)

            if match_exists(session, date, equipe_dom.id, equipe_ext.id):
                stats["matchs_skipped"] += 1
                continue

            # Scores
            score_dom = int(row["FTHG"]) if "FTHG" in row and pd.notna(row["FTHG"]) else None
            score_ext = int(row["FTAG"]) if "FTAG" in row and pd.notna(row["FTAG"]) else None

            match = Match(
                date=date,
                equipe_dom_id=equipe_dom.id,
                equipe_ext_id=equipe_ext.id,
                competition_id=competition.id,
                score_dom=score_dom,
                score_ext=score_ext,
                resultat=ftr_to_resultat(ftr),
                saison=saison,
            )
            session.add(match)
            session.flush()

            # Cotes moyennes
            cote_1 = mean_odds(row, ODDS_COLUMNS_HOME)
            cote_n = mean_odds(row, ODDS_COLUMNS_DRAW)
            cote_2 = mean_odds(row, ODDS_COLUMNS_AWAY)

            if cote_1 or cote_n or cote_2:
                cote = Cote(
                    match_id=match.id,
                    cote_1=cote_1,
                    cote_n=cote_n,
                    cote_2=cote_2,
                    bookmaker="moyenne",
                    date_releve=date,
                )
                session.add(cote)

            stats["matchs_inserted"] += 1

        if own_session:
            session.commit()

    except Exception as e:
        logger.error(f"Erreur lors de l'import de {filepath} : {e}")
        if own_session:
            session.rollback()
        stats["errors"] += 1
    finally:
        if own_session:
            session.close()

    return stats


def import_all(directory: str | None = None) -> dict:
    """Importe tous les CSV du répertoire en base.

    Returns:
        Dict avec les totaux : matchs_inserted, matchs_skipped, errors, files_processed.
    """
    directory = directory or RAW_FOOTBALL_DIR
    totals = {"matchs_inserted": 0, "matchs_skipped": 0, "errors": 0, "files_processed": 0}

    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csv_files:
        logger.warning(f"Aucun CSV trouvé dans {directory}")
        return totals

    init_db()
    session = SessionLocal()

    try:
        for filepath in csv_files:
            logger.info(f"Import de {os.path.basename(filepath)}...")
            result = import_csv(filepath, session=session)
            for key in ["matchs_inserted", "matchs_skipped", "errors"]:
                totals[key] += result[key]
            totals["files_processed"] += 1

        session.commit()
        logger.info(
            f"Import terminé : {totals['files_processed']} fichiers, "
            f"{totals['matchs_inserted']} matchs insérés, "
            f"{totals['matchs_skipped']} ignorés, "
            f"{totals['errors']} erreurs"
        )
    except Exception as e:
        session.rollback()
        logger.error(f"Erreur globale d'import : {e}")
    finally:
        session.close()

    return totals


if __name__ == "__main__":
    logger.info("Démarrage de l'import en base de données")
    result = import_all()
    print(f"Résultat : {result}")
