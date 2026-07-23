"""Dépendances FastAPI (session DB, predictor singleton)."""

from models.predictor import Predictor
from database.connection import SessionLocal


predictor_instance: Predictor = None


def get_predictor() -> Predictor:
    """Singleton Predictor chargé au démarrage."""
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = Predictor()
    return predictor_instance


def get_db():
    """Session de base de données."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
