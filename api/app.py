"""Application FastAPI — API REST LotoFoot IA."""

from fastapi import FastAPI, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import (
    MatchInput, PredictionResponse, BatchPredictRequest,
    GridGenerateRequest, GenerateResponse, GridResponse,
)
from api.deps import get_predictor, get_db
from models.odds_predictor import OddsPredictor
from generator.grid_generator import GridGenerator
from database.models import GrilleLotoFoot, StatistiqueGrille


app = FastAPI(title="LotoFoot IA", version="2.0.0")


@app.get("/api/health")
def health_check():
    """Health check."""
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict_match(
    match: MatchInput,
    predictor: OddsPredictor = Depends(get_predictor),
):
    """Prédire un match à partir des cotes."""
    match_data = {
        "cote_1": getattr(match, "cote_1", 0),
        "cote_n": getattr(match, "cote_n", 0),
        "cote_2": getattr(match, "cote_2", 0),
        "pct_1": getattr(match, "pct_1", 0),
        "pct_n": getattr(match, "pct_n", 0),
        "pct_2": getattr(match, "pct_2", 0),
        "prono_cyborg": getattr(match, "prono_cyborg", ""),
    }

    result = predictor.predict_match(match_data)

    return PredictionResponse(
        prob_1=result["prob_1"],
        prob_n=result["prob_n"],
        prob_2=result["prob_2"],
        prediction=result["prediction"],
        confiance=result["confiance"],
    )


@app.post("/api/predict/batch")
def predict_batch(
    request: BatchPredictRequest,
    predictor: OddsPredictor = Depends(get_predictor),
):
    """Prédire plusieurs matchs."""
    results = []
    for match in request.matches:
        match_data = {
            "cote_1": getattr(match, "cote_1", 0),
            "cote_n": getattr(match, "cote_n", 0),
            "cote_2": getattr(match, "cote_2", 0),
        }
        result = predictor.predict_match(match_data)
        results.append({
            "prob_1": result["prob_1"],
            "prob_n": result["prob_n"],
            "prob_2": result["prob_2"],
            "prediction": result["prediction"],
            "confiance": result["confiance"],
        })

    return results


@app.post("/api/grilles/generate", response_model=GenerateResponse)
def generate_grids(
    request: GridGenerateRequest,
    predictor: OddsPredictor = Depends(get_predictor),
):
    """Générer des grilles optimisées."""
    matches = []
    for match in request.matches:
        matches.append({
            "cote_1": getattr(match, "cote_1", 0),
            "cote_n": getattr(match, "cote_n", 0),
            "cote_2": getattr(match, "cote_2", 0),
            "pct_1": getattr(match, "pct_1", 0),
            "pct_n": getattr(match, "pct_n", 0),
            "pct_2": getattr(match, "pct_2", 0),
            "prono_cyborg": getattr(match, "prono_cyborg", ""),
        })

    grid_data = {"matches": matches}
    generator = GridGenerator(predictor=predictor, strategy=request.strategy)
    grilles = generator.generate(
        grid_data,
        grid_type=request.grid_type,
        budget=request.budget,
    )

    if grilles:
        confiance_moyenne = sum(g["confiance"] for g in grilles) / len(grilles)
    else:
        confiance_moyenne = 0.0

    return GenerateResponse(
        grilles=[GridResponse(**g) for g in grilles],
        stats={
            "nb_grilles": len(grilles),
            "confiance_moyenne": round(confiance_moyenne, 4),
        },
    )


@app.get("/api/grilles/history")
def grilles_history(
    type_grille: str = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
):
    """Historique des grilles réelles."""
    stmt = select(GrilleLotoFoot)

    if type_grille:
        stmt = stmt.where(GrilleLotoFoot.type_grille == type_grille)

    stmt = stmt.order_by(GrilleLotoFoot.date.desc()).limit(limit)
    grilles = session.execute(stmt).scalars().all()

    results = []
    for g in grilles:
        stat = session.execute(
            select(StatistiqueGrille)
            .where(StatistiqueGrille.grille_id == g.id)
        ).scalar()

        entry = {
            "id": g.id,
            "date": str(g.date),
            "type_grille": g.type_grille,
            "resultats": g.resultats,
            "rapport_rang1": g.rapport_rang1,
            "nombre_gagnants_rang1": g.nombre_gagnants_rang1,
        }

        if stat:
            entry["stats"] = {
                "nombre_1": stat.nombre_1,
                "nombre_n": stat.nombre_n,
                "nombre_2": stat.nombre_2,
                "profil": stat.profil,
                "entropie": stat.entropie,
                "indice_chaos": stat.indice_chaos,
            }

        results.append(entry)

    return results


@app.get("/api/stats/distribution")
def stats_distribution():
    """Distribution 1/N/2 par type de grille."""
    try:
        from analysis.grid_analysis import get_distribution_by_type
        return get_distribution_by_type()
    except Exception:
        return {}
