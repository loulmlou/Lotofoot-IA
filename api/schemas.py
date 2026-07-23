"""Schémas Pydantic pour validation des requêtes/réponses de l'API."""

from pydantic import BaseModel, Field


class MatchInput(BaseModel):
    equipe_dom_id: int
    equipe_ext_id: int
    competition_id: int
    date: str


class PredictionResponse(BaseModel):
    prob_1: float
    prob_n: float
    prob_2: float
    prediction: str
    confiance: float


class BatchPredictRequest(BaseModel):
    matches: list[MatchInput]


class GridGenerateRequest(BaseModel):
    matches: list[MatchInput]
    grid_type: str = "LF7"
    budget: int = Field(default=5, ge=1, le=50)
    strategy: str = "equilibree"


class GridResponse(BaseModel):
    resultats: str
    confiance: float
    probabilite: float
    matchs: list[dict]


class GenerateResponse(BaseModel):
    grilles: list[GridResponse]
    stats: dict
