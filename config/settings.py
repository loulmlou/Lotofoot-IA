import os
from dotenv import load_dotenv

load_dotenv()


# === Base de données ===
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/lotofoot.db")

# === API Football ===
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "")
FOOTBALL_API_URL = os.getenv("FOOTBALL_API_URL", "https://v3.football.api-sports.io")

# === Loto Foot ===
FDJ_BASE_URL = "https://www.fdj.fr"

# === Scoring (poids du moteur de décision) ===
SCORING_WEIGHTS = {
    "cotes": 0.30,
    "forme": 0.20,
    "classement": 0.15,
    "historique": 0.10,
    "stats_lotofoot": 0.10,
    "modele_ia": 0.10,
    "contexte": 0.05,
}

# === Budgets disponibles ===
BUDGETS = [5, 10, 15, 20, 50]

# === Stratégies ===
STRATEGIES = ["prudente", "equilibree", "audacieuse"]

# === Football-data.co.uk ===
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

# Saisons de 2010/11 à 2025/26 — format utilisé par football-data.co.uk
FOOTBALL_DATA_SEASONS = [
    "1011", "1112", "1213", "1314", "1415", "1516",
    "1617", "1718", "1819", "1920", "2021", "2122",
    "2223", "2324", "2425", "2526",
]

# === Loto Foot (Parions Sport) ===
LOTOFOOT_BASE_URL = "https://www.pointdevente.parionssport.fdj.fr"
LOTOFOOT_TYPES = {
    "loto-foot-7": {"code": "LF7", "nb_matchs": 7, "id_max_approx": 4910},
    "loto-foot-8": {"code": "LF8", "nb_matchs": 8, "id_max_approx": 795},
    "loto-foot-15": {"code": "LF15", "nb_matchs": 15, "id_max_approx": 2005},
    "loto-foot-12": {"code": "LF12", "nb_matchs": 12, "id_max_approx": 555},
}

# === Chemins de stockage ===
RAW_FOOTBALL_DIR = os.path.join("data", "raw", "football")

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
