"""Dépendances FastAPI (session DB, predictor singleton)."""

from models.odds_predictor import OddsPredictor
from database.connection import SessionLocal


predictor_instance: OddsPredictor = None


def get_predictor() -> OddsPredictor:
    """Singleton OddsPredictor chargé au démarrage."""
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = OddsPredictor()
    return predictor_instance


def get_db():
    """Session de base de données."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
