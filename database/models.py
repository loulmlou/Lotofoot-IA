"""Modèles SQLAlchemy pour la base de données LotoFoot."""

from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, ForeignKey, Enum, Text,
    create_engine
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class Equipe(Base):
    __tablename__ = "equipes"

    id = Column(Integer, primary_key=True)
    nom = Column(String(100), nullable=False)
    pays = Column(String(50), nullable=False)
    competition = Column(String(100))
    logo_url = Column(String(255))
    api_id = Column(Integer, unique=True)  # ID dans l'API Football

    matchs_domicile = relationship("Match", foreign_keys="Match.equipe_dom_id", back_populates="equipe_dom")
    matchs_exterieur = relationship("Match", foreign_keys="Match.equipe_ext_id", back_populates="equipe_ext")


class Competition(Base):
    __tablename__ = "competitions"

    id = Column(Integer, primary_key=True)
    nom = Column(String(100), nullable=False)
    pays = Column(String(50))
    saison = Column(String(10))
    api_id = Column(Integer)


class Match(Base):
    __tablename__ = "matchs"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    equipe_dom_id = Column(Integer, ForeignKey("equipes.id"), nullable=False)
    equipe_ext_id = Column(Integer, ForeignKey("equipes.id"), nullable=False)
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    score_dom = Column(Integer)
    score_ext = Column(Integer)
    resultat = Column(String(1))  # '1', 'N', '2'
    saison = Column(String(10))
    journee = Column(Integer)

    equipe_dom = relationship("Equipe", foreign_keys=[equipe_dom_id], back_populates="matchs_domicile")
    equipe_ext = relationship("Equipe", foreign_keys=[equipe_ext_id], back_populates="matchs_exterieur")
    competition = relationship("Competition")
    cotes = relationship("Cote", back_populates="match", uselist=False)


class Cote(Base):
    __tablename__ = "cotes"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matchs.id"), nullable=False, unique=True)
    cote_1 = Column(Float)
    cote_n = Column(Float)
    cote_2 = Column(Float)
    bookmaker = Column(String(50))
    date_releve = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="cotes")


class GrilleLotoFoot(Base):
    __tablename__ = "grilles_lotofoot"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    type_grille = Column(String(20), nullable=False)  # 'LF7', 'LF8', 'LF15'
    resultats = Column(String(20))  # ex: '1N21N121'
    rapport_rang1 = Column(Float)
    nombre_gagnants_rang1 = Column(Integer)
    mise_totale = Column(Float)

    matchs_grille = relationship("MatchGrille", back_populates="grille")


class MatchGrille(Base):
    __tablename__ = "matchs_grille"

    id = Column(Integer, primary_key=True)
    grille_id = Column(Integer, ForeignKey("grilles_lotofoot.id"), nullable=False)
    position = Column(Integer, nullable=False)  # Position 1-8 dans la grille
    match_id = Column(Integer, ForeignKey("matchs.id"))
    resultat = Column(String(1))  # '1', 'N', '2'

    grille = relationship("GrilleLotoFoot", back_populates="matchs_grille")
    match = relationship("Match")


class StatistiqueGrille(Base):
    __tablename__ = "statistiques_grille"

    id = Column(Integer, primary_key=True)
    grille_id = Column(Integer, ForeignKey("grilles_lotofoot.id"), nullable=False, unique=True)
    nombre_1 = Column(Integer, default=0)
    nombre_n = Column(Integer, default=0)
    nombre_2 = Column(Integer, default=0)
    indice_chaos = Column(Float)
    entropie = Column(Float)
    alternance = Column(Float)
    profil = Column(String(10))  # ex: '5-2-1'
    plus_longue_suite = Column(Integer)

    grille = relationship("GrilleLotoFoot")
