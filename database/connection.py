"""Gestion de la connexion à la base de données."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import DATABASE_URL
from database.models import Base


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Créer toutes les tables dans la base de données."""
    Base.metadata.create_all(engine)


def get_session():
    """Retourne une session de base de données."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
