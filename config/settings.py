import os
from dotenv import load_dotenv

load_dotenv()


# === Base de données ===
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/lotofoot.db")

# === Loto Foot ===
FDJ_BASE_URL = "https://www.fdj.fr"

# === Budgets disponibles ===
BUDGETS = [5, 10, 15, 20, 50]

# === Stratégies ===
STRATEGIES = ["prudente", "equilibree", "audacieuse"]

# === Loto Foot (Parions Sport) ===
LOTOFOOT_BASE_URL = "https://www.pointdevente.parionssport.fdj.fr"
LOTOFOOT_TYPES = {
    "loto-foot-7": {"code": "LF7", "nb_matchs": 7, "id_max_approx": 4910},
    "loto-foot-8": {"code": "LF8", "nb_matchs": 8, "id_max_approx": 795},
    "loto-foot-15": {"code": "LF15", "nb_matchs": 15, "id_max_approx": 2005},
    "loto-foot-12": {"code": "LF12", "nb_matchs": 12, "id_max_approx": 555},
}

# === Modèle basé sur les cotes (OddsPredictor) ===
ODDS_MODEL_WEIGHTS = {
    "cotes": 0.60,
    "consensus": 0.15,
    "cyborg": 0.10,
    "difficulte": 0.10,
    "historique": 0.05,
}

# === Seuils de difficulté pour la stratégie adaptative ===
DIFFICULTY_THRESHOLDS = {
    "secure_max_difficulty": 5.0,
    "secure_max_cote_fav": 1.70,
    "risky_min_difficulty": 8.0,
    "risky_min_cote_fav": 2.20,
}

# === Football-data.co.uk (legacy, utilisé par collectors/football_data.py) ===
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"

FOOTBALL_DATA_LEAGUES = {
    "F1": {"nom": "Ligue 1", "pays": "France"},
    "F2": {"nom": "Ligue 2", "pays": "France"},
    "E0": {"nom": "Premier League", "pays": "Angleterre"},
    "E1": {"nom": "Championship", "pays": "Angleterre"},
    "SP1": {"nom": "La Liga", "pays": "Espagne"},
    "SP2": {"nom": "Segunda División", "pays": "Espagne"},
    "I1": {"nom": "Serie A", "pays": "Italie"},
    "I2": {"nom": "Serie B", "pays": "Italie"},
    "D1": {"nom": "Bundesliga", "pays": "Allemagne"},
    "D2": {"nom": "2. Bundesliga", "pays": "Allemagne"},
    "N1": {"nom": "Eredivisie", "pays": "Pays-Bas"},
    "P1": {"nom": "Primeira Liga", "pays": "Portugal"},
    "B1": {"nom": "Jupiler League", "pays": "Belgique"},
    "T1": {"nom": "Süper Lig", "pays": "Turquie"},
}

FOOTBALL_DATA_SEASONS = [
    "1011", "1112", "1213", "1314", "1415", "1516",
    "1617", "1718", "1819", "1920", "2021", "2122",
    "2223", "2324", "2425", "2526",
]

# === The Odds API (fallback cotes manquantes) ===
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# === Chemins de stockage ===
RAW_FOOTBALL_DIR = os.path.join("data", "raw", "football")
HISTORY_DIR = os.path.join("data", "history")

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
