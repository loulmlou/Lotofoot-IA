"""Tests pour le module d'analyse (Phase 2)."""

import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Match, Equipe, Competition, Cote, GrilleLotoFoot, MatchGrille, StatistiqueGrille


@pytest.fixture
def db_session():
    """Crée une base de données en mémoire avec des données de test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # --- Compétitions ---
    comp = Competition(id=1, nom="Ligue 1", pays="France", saison="2023/2024")
    session.add(comp)

    # --- Equipes ---
    psg = Equipe(id=1, nom="PSG", pays="France")
    om = Equipe(id=2, nom="OM", pays="France")
    lyon = Equipe(id=3, nom="Lyon", pays="France")
    session.add_all([psg, om, lyon])

    # --- Matchs (20 matchs pour avoir assez d'historique) ---
    matchs_data = [
        # PSG dom vs OM ext — PSG gagne
        (1, datetime(2023, 8, 15), 1, 2, 1, 3, 1, "1", "2023/2024", 1),
        # OM dom vs Lyon ext — Nul
        (2, datetime(2023, 8, 20), 2, 3, 1, 1, 1, "N", "2023/2024", 1),
        # Lyon dom vs PSG ext — PSG gagne (ext)
        (3, datetime(2023, 8, 25), 3, 1, 1, 0, 2, "2", "2023/2024", 1),
        # PSG dom vs Lyon ext — PSG gagne
        (4, datetime(2023, 9, 1), 1, 3, 1, 2, 0, "1", "2023/2024", 2),
        # OM dom vs PSG ext — PSG gagne (ext)
        (5, datetime(2023, 9, 10), 2, 1, 1, 0, 1, "2", "2023/2024", 2),
        # Lyon dom vs OM ext — Lyon gagne
        (6, datetime(2023, 9, 15), 3, 2, 1, 2, 1, "1", "2023/2024", 2),
        # PSG dom vs OM ext — Nul
        (7, datetime(2023, 9, 20), 1, 2, 1, 1, 1, "N", "2023/2024", 3),
        # OM dom vs Lyon ext — OM gagne
        (8, datetime(2023, 9, 25), 2, 3, 1, 3, 0, "1", "2023/2024", 3),
        # Lyon dom vs PSG ext — Nul
        (9, datetime(2023, 10, 1), 3, 1, 1, 2, 2, "N", "2023/2024", 3),
        # PSG dom vs Lyon ext — PSG gagne
        (10, datetime(2023, 10, 5), 1, 3, 1, 4, 0, "1", "2023/2024", 4),
        # Matchs supplémentaires pour atteindre min_history
        (11, datetime(2023, 10, 10), 1, 2, 1, 2, 0, "1", "2023/2024", 4),
        (12, datetime(2023, 10, 15), 2, 3, 1, 1, 1, "N", "2023/2024", 4),
        (13, datetime(2023, 10, 20), 3, 1, 1, 1, 3, "2", "2023/2024", 5),
        (14, datetime(2023, 10, 25), 1, 3, 1, 2, 1, "1", "2023/2024", 5),
        (15, datetime(2023, 10, 30), 2, 1, 1, 0, 2, "2", "2023/2024", 5),
    ]

    for mid, dt, dom_id, ext_id, comp_id, sd, se, res, saison, j in matchs_data:
        m = Match(
            id=mid, date=dt, equipe_dom_id=dom_id, equipe_ext_id=ext_id,
            competition_id=comp_id, score_dom=sd, score_ext=se,
            resultat=res, saison=saison, journee=j,
        )
        session.add(m)

    # --- Cotes ---
    cotes_data = [
        (1, 1, 1.5, 4.0, 6.0),
        (2, 2, 2.0, 3.2, 3.5),
        (3, 3, 2.5, 3.0, 2.8),
        (4, 4, 1.3, 5.0, 8.0),
        (5, 5, 1.8, 3.5, 4.0),
        (6, 6, 2.2, 3.3, 3.0),
        (7, 7, 1.6, 3.8, 5.5),
        (8, 8, 1.9, 3.4, 3.8),
        (9, 9, 3.0, 3.2, 2.3),
        (10, 10, 1.4, 4.5, 7.0),
        (11, 11, 1.5, 4.0, 6.0),
        (12, 12, 2.0, 3.2, 3.5),
        (13, 13, 2.5, 3.0, 2.8),
        (14, 14, 1.3, 5.0, 8.0),
        (15, 15, 1.8, 3.5, 4.0),
    ]

    for cid, mid, c1, cn, c2 in cotes_data:
        c = Cote(id=cid, match_id=mid, cote_1=c1, cote_n=cn, cote_2=c2, bookmaker="moyenne")
        session.add(c)

    # --- Grille Loto Foot ---
    grille = GrilleLotoFoot(
        id=1, date=date(2023, 10, 1), type_grille="LF7", resultats="1N21N12",
        rapport_rang1=50000.0, nombre_gagnants_rang1=2, mise_totale=100000.0,
    )
    session.add(grille)

    grille2 = GrilleLotoFoot(
        id=2, date=date(2023, 10, 15), type_grille="LF7", resultats="1111111",
        rapport_rang1=10000.0, nombre_gagnants_rang1=10, mise_totale=80000.0,
    )
    session.add(grille2)

    # MatchGrille pour grille 1
    for i, res in enumerate("1N21N12"):
        mg = MatchGrille(
            grille_id=1, position=i + 1,
            match_id=i + 1 if i + 1 <= 7 else None,
            resultat=res,
        )
        session.add(mg)

    session.commit()
    yield session
    session.close()


# =====================================================
# Tests des features (analysis/features.py)
# =====================================================

class TestComputeForm:
    def test_basic_form(self, db_session):
        from analysis.features import compute_form

        # Forme du PSG avant le match du 2023-10-05
        result = compute_form(1, datetime(2023, 10, 5), n=5, session=db_session)

        assert result["nb_matchs"] == 5
        assert result["points"] >= 0
        assert 0 <= result["ratio_victoires"] <= 1
        assert result["buts_marques"] >= 0

    def test_no_history(self, db_session):
        from analysis.features import compute_form

        # Pas d'historique avant le tout premier match
        result = compute_form(1, datetime(2023, 1, 1), n=5, session=db_session)
        assert result["nb_matchs"] == 0
        assert result["points"] == 0

    def test_form_home_away_split(self, db_session):
        from analysis.features import compute_form

        result = compute_form(1, datetime(2023, 11, 1), n=10, session=db_session)
        # PSG a joué des matchs à domicile et à l'extérieur
        assert result["forme_dom"]["nb"] + result["forme_ext"]["nb"] == result["nb_matchs"]


class TestComputeH2H:
    def test_h2h_psg_om(self, db_session):
        from analysis.features import compute_h2h

        result = compute_h2h(1, 2, datetime(2023, 11, 1), n=5, session=db_session)
        assert result["nb_matchs"] > 0
        assert abs(result["pct_dom"] + result["pct_nul"] + result["pct_ext"] - 1.0) < 0.01

    def test_h2h_no_history(self, db_session):
        from analysis.features import compute_h2h

        # Equipes sans confrontation
        result = compute_h2h(1, 99, datetime(2023, 11, 1), n=5, session=db_session)
        assert result["nb_matchs"] == 0


class TestComputeStanding:
    def test_standing_psg(self, db_session):
        from analysis.features import compute_standing

        result = compute_standing(1, 1, datetime(2023, 10, 5), session=db_session)
        assert result["position"] is not None
        assert result["position"] >= 1
        assert result["points"] > 0

    def test_standing_no_data(self, db_session):
        from analysis.features import compute_standing

        result = compute_standing(1, 99, datetime(2023, 1, 1), session=db_session)
        assert result["position"] is None


class TestComputeGoalStats:
    def test_goal_stats(self, db_session):
        from analysis.features import compute_goal_stats

        result = compute_goal_stats(1, datetime(2023, 11, 1), n=10, session=db_session)
        assert result["nb_matchs"] > 0
        assert result["moy_buts_marques"] > 0
        assert 0 <= result["over_2_5_ratio"] <= 1


class TestComputeOddsFeatures:
    def test_odds_with_cotes(self):
        from analysis.features import compute_odds_features

        cote = {"cote_1": 1.5, "cote_n": 4.0, "cote_2": 6.0}
        result = compute_odds_features(cote)

        assert result["prob_1"] is not None
        assert abs(result["prob_1"] + result["prob_n"] + result["prob_2"] - 1.0) < 0.001
        assert result["prob_1"] > result["prob_n"] > result["prob_2"]

    def test_odds_none(self):
        from analysis.features import compute_odds_features

        result = compute_odds_features(None)
        assert result["prob_1"] is None


class TestBuildMatchFeatures:
    def test_build_features(self, db_session):
        from analysis.features import build_match_features

        # Match avec suffisamment d'historique
        match = db_session.get(Match, 10)
        features = build_match_features(match, session=db_session)

        assert "dom_forme_pts" in features
        assert "ext_forme_pts" in features
        assert "h2h_nb" in features
        assert "prob_1" in features
        assert "dom_jours_repos" in features
        assert "avantage_dom_ligue" in features


# =====================================================
# Tests des stats de grilles (analysis/grid_analysis.py)
# =====================================================

class TestGridAnalysis:
    def test_entropy(self):
        from analysis.grid_analysis import _compute_entropy

        # Entropie max pour 3 symboles équiprobables
        assert _compute_entropy("1N2") > 0
        # Entropie 0 pour séquence uniforme
        assert _compute_entropy("1111") == 0.0
        # Entropie("1N2") devrait être ~log2(3) ≈ 1.585
        assert abs(_compute_entropy("1N2") - math.log2(3)) < 0.01

    def test_alternance(self):
        from analysis.grid_analysis import _compute_alternance

        # Alternance max
        assert _compute_alternance("1N1N1N1") == 1.0
        # Alternance 0
        assert _compute_alternance("1111111") == 0.0

    def test_plus_longue_suite(self):
        from analysis.grid_analysis import _compute_plus_longue_suite

        assert _compute_plus_longue_suite("1111N22") == 4
        assert _compute_plus_longue_suite("1N2") == 1

    def test_indice_chaos_heuristique(self):
        from analysis.grid_analysis import _compute_indice_chaos

        # Sans cotes, N et 2 comptent comme surprenants
        chaos = _compute_indice_chaos("1N21N12")
        assert 0 < chaos < 1

    def test_compute_grid_stats(self, db_session):
        from analysis.grid_analysis import compute_grid_stats

        grille = db_session.get(GrilleLotoFoot, 1)
        stat = compute_grid_stats(grille, session=db_session)

        # "1N21N12" → trois '1', deux 'N', deux '2'
        assert stat.nombre_1 == 3
        assert stat.nombre_n == 2
        assert stat.nombre_2 == 2
        assert stat.profil == "3-2-2"
        assert stat.entropie > 0
        assert stat.plus_longue_suite >= 1

    def test_compute_grid_stats_uniform(self, db_session):
        from analysis.grid_analysis import compute_grid_stats

        grille = db_session.get(GrilleLotoFoot, 2)
        stat = compute_grid_stats(grille, session=db_session)

        assert stat.nombre_1 == 7
        assert stat.nombre_n == 0
        assert stat.nombre_2 == 0
        assert stat.entropie == 0.0
        assert stat.alternance == 0.0
        assert stat.plus_longue_suite == 7


# =====================================================
# Tests du dataset builder (analysis/dataset_builder.py)
# =====================================================

class TestDatasetBuilder:
    def test_split_by_date(self):
        from analysis.dataset_builder import split_by_date

        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2023-01-01", "2023-06-01", "2023-09-01", "2024-01-01",
            ]),
            "resultat": ["1", "N", "2", "1"],
            "feature_1": [1, 2, 3, 4],
        })

        train, val, test = split_by_date(df, "2023-06-30", "2023-10-01")

        assert len(train) == 2
        assert len(val) == 1
        assert len(test) == 1

    def test_save_load_dataset(self, tmp_path):
        from analysis.dataset_builder import save_dataset, load_dataset

        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-06-01"]),
            "resultat": ["1", "N"],
            "feature_1": [1.0, 2.0],
        })

        path = str(tmp_path / "test_dataset.csv")
        save_dataset(df, path)
        loaded = load_dataset(path)

        assert len(loaded) == 2
        assert list(loaded.columns) == list(df.columns)


# =====================================================
# Tests des baselines (analysis/baseline.py)
# =====================================================

class TestBaselines:
    @pytest.fixture
    def sample_df(self):
        """DataFrame de test avec résultats et cotes."""
        np.random.seed(42)
        n = 100
        resultats = np.random.choice(["1", "N", "2"], size=n, p=[0.44, 0.27, 0.29])
        return pd.DataFrame({
            "resultat": resultats,
            "cote_1": np.random.uniform(1.2, 5.0, n),
            "cote_n": np.random.uniform(2.5, 5.0, n),
            "cote_2": np.random.uniform(1.5, 8.0, n),
            "prob_1": np.random.uniform(0.2, 0.7, n),
            "prob_n": np.random.uniform(0.1, 0.4, n),
            "prob_2": np.random.uniform(0.1, 0.5, n),
        })

    def test_baseline_random(self, sample_df):
        from analysis.baseline import baseline_random

        result = baseline_random(sample_df)
        assert result["name"] == "random"
        assert 0 < result["accuracy"] < 1
        assert "log_loss" in result

    def test_baseline_home(self, sample_df):
        from analysis.baseline import baseline_home

        result = baseline_home(sample_df)
        assert result["name"] == "always_home"
        # Accuracy devrait correspondre au % de victoires domicile
        pct_1 = (sample_df["resultat"] == "1").mean()
        assert abs(result["accuracy"] - pct_1) < 0.01

    def test_baseline_odds_favorite(self, sample_df):
        from analysis.baseline import baseline_odds_favorite

        result = baseline_odds_favorite(sample_df)
        assert result["name"] == "odds_favorite"
        assert 0 < result["accuracy"] < 1
        assert "log_loss" in result

    def test_baseline_distribution(self, sample_df):
        from analysis.baseline import baseline_distribution

        result = baseline_distribution(sample_df)
        assert result["name"] == "distribution"
        assert 0 < result["accuracy"] < 1

    def test_evaluate_predictions_accuracy(self):
        from analysis.baseline import evaluate_predictions

        y_true = ["1", "N", "2", "1", "1"]
        y_pred = ["1", "N", "1", "1", "2"]

        result = evaluate_predictions(y_true, y_pred)
        assert result["accuracy"] == 3 / 5

    def test_evaluate_predictions_roi(self):
        from analysis.baseline import evaluate_predictions

        y_true = ["1", "N", "2"]
        y_pred = ["1", "N", "2"]
        cotes = [
            {"cote_1": 2.0, "cote_n": 3.0, "cote_2": 4.0},
            {"cote_1": 2.0, "cote_n": 3.0, "cote_2": 4.0},
            {"cote_1": 2.0, "cote_n": 3.0, "cote_2": 4.0},
        ]

        result = evaluate_predictions(y_true, y_pred, cotes=cotes)
        # Gains: 2.0 + 3.0 + 4.0 = 9.0, Mise: 3, ROI = (9-3)/3 = 2.0
        assert result["accuracy"] == 1.0
        assert abs(result["roi"] - 2.0) < 0.01

    def test_baseline_scores_in_bounds(self, sample_df):
        from analysis.baseline import baseline_random, baseline_home

        random_result = baseline_random(sample_df)
        home_result = baseline_home(sample_df)

        # Random devrait être autour de 33%
        assert 0.15 < random_result["accuracy"] < 0.55
        # Log loss devrait être positif
        assert random_result["log_loss"] > 0
        # Home devrait être autour de 44%
        assert 0.25 < home_result["accuracy"] < 0.65
