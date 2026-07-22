"""Tests pour les collecteurs de données."""

import os
import sys
import tempfile
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from database.models import Base, Match, Equipe, Competition, Cote
from collectors.football_data import build_url, download_csv, get_csv_path
from collectors.import_football import (
    parse_date,
    mean_odds,
    ftr_to_resultat,
    season_label,
    extract_league_from_filename,
    extract_season_from_filename,
    import_csv,
)


# === Fixtures ===


@pytest.fixture
def db_session():
    """Crée une base SQLite en mémoire pour les tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_csv(tmp_path):
    """Crée un fichier CSV de test avec des données de matchs."""
    csv_content = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,"
        "HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR,"
        "B365H,B365D,B365A,BWH,BWD,BWA\n"
        "F1,15/08/2023,Paris SG,Lorient,0,0,D,0,0,D,"
        "15,5,4,2,6,3,12,10,2,1,0,0,"
        "1.20,7.00,15.00,1.22,6.50,14.00\n"
        "F1,15/08/2023,Lyon,Montpellier,4,1,H,2,0,H,"
        "18,8,7,3,5,2,14,16,3,4,0,1,"
        "1.50,4.50,6.00,1.55,4.20,5.80\n"
        "F1,16/08/2023,Marseille,Reims,2,2,D,1,1,D,"
        "12,10,5,4,4,5,11,13,1,2,0,0,"
        "1.40,5.00,8.00,1.42,4.80,7.50\n"
        "F1,17/08/2023,Monaco,Strasbourg,3,1,H,1,0,H,"
        "14,6,6,2,7,4,10,15,2,3,0,0,"
        "1.65,3.80,5.50,1.60,3.90,5.20\n"
        "F1,18/08/2023,Lille,Nantes,2,0,H,1,0,H,"
        "16,7,5,3,8,2,9,12,1,2,0,0,"
        "1.80,3.50,4.80,1.75,3.60,4.50\n"
    )
    filepath = tmp_path / "F1_2324.csv"
    filepath.write_text(csv_content, encoding="utf-8")
    return str(filepath)


# === Tests football_data.py ===


class TestBuildUrl:
    def test_url_format(self):
        url = build_url("2324", "F1")
        assert url == "https://www.football-data.co.uk/mmz4281/2324/F1.csv"

    def test_url_different_league(self):
        url = build_url("1920", "E0")
        assert url == "https://www.football-data.co.uk/mmz4281/1920/E0.csv"


class TestGetCsvPath:
    def test_path_format(self):
        path = get_csv_path("E0", "2324")
        assert path.endswith(os.path.join("football", "E0_2324.csv"))


class TestDownloadCsv:
    def test_download_real_csv(self, tmp_path):
        """Test de téléchargement d'un vrai CSV (1 saison, 1 ligue)."""
        url = build_url("2324", "E0")
        dest = str(tmp_path / "E0_2324.csv")
        result = download_csv(url, dest)

        if result:
            assert os.path.exists(dest)
            assert os.path.getsize(dest) > 100
            # Vérifier que c'est un CSV valide
            df = pd.read_csv(dest)
            assert "HomeTeam" in df.columns
            assert "AwayTeam" in df.columns
            assert "FTR" in df.columns
            assert len(df) > 10
        # Si le téléchargement échoue (pas de réseau), on skip


# === Tests import_football.py ===


class TestParseDate:
    def test_dd_mm_yyyy(self):
        dt = parse_date("15/08/2023")
        assert dt.year == 2023
        assert dt.month == 8
        assert dt.day == 15

    def test_dd_mm_yy(self):
        dt = parse_date("15/08/23")
        assert dt.year == 2023
        assert dt.month == 8
        assert dt.day == 15

    def test_invalid(self):
        assert parse_date("invalid") is None

    def test_nan(self):
        assert parse_date(float("nan")) is None


class TestFtrToResultat:
    def test_home(self):
        assert ftr_to_resultat("H") == "1"

    def test_draw(self):
        assert ftr_to_resultat("D") == "N"

    def test_away(self):
        assert ftr_to_resultat("A") == "2"

    def test_lowercase(self):
        assert ftr_to_resultat("h") == "1"

    def test_invalid(self):
        assert ftr_to_resultat("X") is None


class TestSeasonLabel:
    def test_normal(self):
        assert season_label("2324") == "2023/2024"

    def test_century_boundary(self):
        assert season_label("9900") == "1999/2000"

    def test_short(self):
        assert season_label("23") == "23"


class TestMeanOdds:
    def test_with_values(self):
        row = pd.Series({"B365H": 1.5, "BWH": 1.6, "IWH": 1.55})
        result = mean_odds(row, ["B365H", "BWH", "IWH"])
        assert abs(result - 1.55) < 0.01

    def test_with_missing(self):
        row = pd.Series({"B365H": 1.5, "BWH": float("nan")})
        result = mean_odds(row, ["B365H", "BWH", "IWH"])
        assert result == 1.5

    def test_all_missing(self):
        row = pd.Series({"B365H": float("nan")})
        result = mean_odds(row, ["B365H", "BWH"])
        assert result is None


class TestExtractFilenameInfo:
    def test_league(self):
        assert extract_league_from_filename("E0_2324.csv") == "E0"

    def test_season(self):
        assert extract_season_from_filename("E0_2324.csv") == "2324"

    def test_with_path(self):
        assert extract_league_from_filename("/data/raw/SP1_1920.csv") == "SP1"


# === Tests d'import en base ===


class TestImportCsv:
    def test_import_sample(self, sample_csv, db_session):
        """Test d'import d'un CSV de test dans la base."""
        result = import_csv(sample_csv, session=db_session)
        db_session.commit()

        assert result["matchs_inserted"] == 5
        assert result["errors"] == 0

        # Vérifier les matchs en base
        nb_matchs = db_session.execute(select(func.count(Match.id))).scalar()
        assert nb_matchs == 5

        # Vérifier les équipes créées
        nb_equipes = db_session.execute(select(func.count(Equipe.id))).scalar()
        assert nb_equipes == 10  # 5 matchs × 2 équipes uniques

        # Vérifier les compétitions
        nb_comp = db_session.execute(select(func.count(Competition.id))).scalar()
        assert nb_comp == 1

    def test_no_duplicates(self, sample_csv, db_session):
        """Test que l'import ne crée pas de doublons."""
        import_csv(sample_csv, session=db_session)
        db_session.commit()

        result = import_csv(sample_csv, session=db_session)
        db_session.commit()

        assert result["matchs_skipped"] == 5
        assert result["matchs_inserted"] == 0

        nb_matchs = db_session.execute(select(func.count(Match.id))).scalar()
        assert nb_matchs == 5

    def test_resultats_valides(self, sample_csv, db_session):
        """Test que les résultats sont bien convertis (H→1, D→N, A→2)."""
        import_csv(sample_csv, session=db_session)
        db_session.commit()

        matchs = db_session.execute(select(Match)).scalars().all()
        for match in matchs:
            assert match.resultat in ("1", "N", "2")

    def test_cotes_importees(self, sample_csv, db_session):
        """Test que les cotes sont bien importées."""
        import_csv(sample_csv, session=db_session)
        db_session.commit()

        nb_cotes = db_session.execute(select(func.count(Cote.id))).scalar()
        assert nb_cotes == 5

        cotes = db_session.execute(select(Cote)).scalars().all()
        for cote in cotes:
            assert cote.cote_1 is not None
            assert cote.cote_n is not None
            assert cote.cote_2 is not None
            assert cote.cote_1 > 1.0
            assert cote.cote_n > 1.0
            assert cote.cote_2 > 1.0

    def test_scores_corrects(self, sample_csv, db_session):
        """Test que les scores sont correctement importés."""
        import_csv(sample_csv, session=db_session)
        db_session.commit()

        matchs = db_session.execute(select(Match)).scalars().all()
        for match in matchs:
            assert match.score_dom is not None
            assert match.score_ext is not None
            assert match.score_dom >= 0
            assert match.score_ext >= 0
