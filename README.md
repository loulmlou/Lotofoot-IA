# LotoFoot AI Analyst

Outil d'analyse et de prediction pour les grilles Loto Foot (FDJ). Combine scraping des grilles historiques, feature engineering, modeles de machine learning et generation de grilles optimisees.

## Architecture

```
LotoFoot-IA/
├── analysis/          # Feature engineering, stats de grilles, baselines
├── api/               # API REST FastAPI
├── backtesting/       # Moteur de backtesting et validation historique
├── collectors/        # Scraping FDJ + import football-data.co.uk
├── config/            # Configuration (settings, strategies, budgets)
├── data/              # Donnees brutes et modeles entraines
├── database/          # Modeles SQLAlchemy et connexion DB
├── frontend/          # Interface Streamlit
├── generator/         # Generateur et optimiseur de grilles
├── models/            # Scoring multi-criteres, Predictor, entrainement ML
├── tests/             # Suite de tests (138 tests)
└── main.py            # Point d'entree CLI
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

Creer un fichier `.env` a la racine :

```
DATABASE_URL=sqlite:///data/lotofoot.db
FOOTBALL_API_KEY=votre_cle  # optionnel
```

Initialiser la base de donnees :

```bash
python main.py
```

## Utilisation

### Interface Streamlit

```bash
python main.py ui
```

L'interface propose 4 pages :

- **Generateur de grilles** : deux modes disponibles
  - **Grille a venir (FDJ)** : recherche automatique des grilles a venir sur le site FDJ, preremplissage des matchs avec matching flou des noms d'equipes, puis generation de predictions en un clic
  - **Saisie manuelle** : selection manuelle des equipes, competition et date pour chaque match
- **Prediction de match** : probabilites 1/N/2 pour un match individuel avec graphiques
- **Historique des grilles** : consultation des grilles passees avec resultats et statistiques
- **Dashboard stats** : distribution 1/N/2 par type, entropie, chaos, alternance

### API REST

```bash
python main.py serve
```

Endpoints disponibles sur `http://localhost:8000` :

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Prediction d'un match |
| POST | `/api/predict/batch` | Prediction de plusieurs matchs |
| POST | `/api/grilles/generate` | Generation de grilles optimisees |
| GET | `/api/grilles/history` | Historique des grilles reelles |
| GET | `/api/stats/distribution` | Distribution 1/N/2 par type |

### Scraping des donnees

```bash
# Scraping des grilles Loto Foot historiques
python -c "from collectors.lotofoot_scraper import scrape_all; scrape_all()"

# Recherche des grilles a venir (sans stockage en DB)
python -c "from collectors.lotofoot_scraper import fetch_upcoming_grilles; print(fetch_upcoming_grilles())"

# Import des donnees football-data.co.uk
python -c "from collectors.football_data import import_all; import_all()"
```

## Moteur de prediction

Le scoring combine plusieurs sources ponderees :

| Source | Poids |
|--------|-------|
| Cotes bookmakers | 30% |
| Forme recente | 20% |
| Classement | 15% |
| Historique H2H | 10% |
| Stats Loto Foot | 10% |
| Modele IA (XGBoost/LightGBM) | 10% |
| Contexte (domicile/exterieur) | 5% |

Trois strategies de generation : **prudente**, **equilibree**, **audacieuse**.

## Types de grilles supportes

| Type | Code | Matchs |
|------|------|--------|
| Loto Foot 7 | LF7 | 7 |
| Loto Foot 8 | LF8 | 8 |
| Loto Foot 12 | LF12 | 12 |
| Loto Foot 15 | LF15 | 15 |

## Tests

```bash
pytest tests/ -v
```

138 tests couvrant l'ensemble des modules : analysis, api, collectors, frontend, generator, models.

## Technologies

- **Backend** : Python 3.13, FastAPI, SQLAlchemy, SQLite
- **ML** : scikit-learn, XGBoost, LightGBM
- **Frontend** : Streamlit, Plotly
- **Scraping** : requests, BeautifulSoup, lxml
- **Tests** : pytest
