"""Application FastAPI — API REST LotoFoot IA."""

from fastapi import FastAPI, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import (
    MatchInput, PredictionResponse, BatchPredictRequest,
    GridGenerateRequest, GenerateResponse, GridResponse,
)
from api.deps import get_predictor, get_db
from models.predictor import Predictor
from generator.grid_generator import GridGenerator
from database.models import GrilleLotoFoot, StatistiqueGrille
from analysis.grid_analysis import get_distribution_by_type


app = FastAPI(title="LotoFoot IA", version="1.0.0")


@app.get("/api/health")
def health_check():
    """Health check."""
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict_match(
    match: MatchInput,
    predictor: Predictor = Depends(get_predictor),
    session: Session = Depends(get_db),
):
    """Prédire un match à partir des IDs des équipes."""
    result = predictor.predict_from_ids(
        equipe_dom_id=match.equipe_dom_id,
        equipe_ext_id=match.equipe_ext_id,
        competition_id=match.competition_id,
        date=match.date,
        session=session,
    )

    if result is None:
        return PredictionResponse(
            prob_1=1 / 3, prob_n=1 / 3, prob_2=1 / 3,
            prediction="1", confiance=0.0,
        )

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
    predictor: Predictor = Depends(get_predictor),
    session: Session = Depends(get_db),
):
    """Prédire plusieurs matchs."""
    results = []
    for match in request.matches:
        result = predictor.predict_from_ids(
            equipe_dom_id=match.equipe_dom_id,
            equipe_ext_id=match.equipe_ext_id,
            competition_id=match.competition_id,
            date=match.date,
            session=session,
        )

        if result is None:
            results.append({
                "prob_1": 1 / 3, "prob_n": 1 / 3, "prob_2": 1 / 3,
                "prediction": "1", "confiance": 0.0,
            })
        else:
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
    predictor: Predictor = Depends(get_predictor),
    session: Session = Depends(get_db),
):
    """Générer des grilles optimisées."""
    # Construire les features pour chaque match
    match_features = []
    for match in request.matches:
        result = predictor.predict_from_ids(
            equipe_dom_id=match.equipe_dom_id,
            equipe_ext_id=match.equipe_ext_id,
            competition_id=match.competition_id,
            date=match.date,
            session=session,
        )
        if result:
            # Utiliser le résultat de predict_from_ids comme features
            match_features.append({
                "prob_1": result["prob_1"],
                "prob_n": result["prob_n"],
                "prob_2": result["prob_2"],
                "prediction": result["prediction"],
                "confiance": result["confiance"],
            })
        else:
            match_features.append({
                "prob_1": 1 / 3, "prob_n": 1 / 3, "prob_2": 1 / 3,
            })

    generator = GridGenerator(predictor=predictor, strategy=request.strategy)
    grilles = generator.generate(
        match_features,
        grid_type=request.grid_type,
        budget=request.budget,
    )

    # Calculer les stats
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
        # Chercher les stats associées
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
    return get_distribution_by_type()
