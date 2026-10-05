"""API de serving du modèle sirtuin6 (FastAPI).

Principes :
- le modèle vient du registre MLflow, jamais d'un fichier local : par défaut la version pointée
  par l'alias `production`, posé à la main après revue ;
- il est chargé une seule fois, au démarrage ; chaque réponse indique la version qui l'a produite ;
- l'API refuse de démarrer si le registre est injoignable ou si le modèle ne respecte pas le
  contrat d'entrée : mieux vaut un service absent qu'un service qui prédit faux.
"""

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from config.ml_params import CLASSES, POSITIVE_CLASS
from config.settings import API_PORT, DEPLOYING_PATH, FRONTEND_URL
from src.deploying.app.model import ServedModel, load_from_registry
from src.deploying.app.schemas import (
    BatchRequest,
    BatchResponse,
    Health,
    ModelInfo,
    ModelRef,
    Molecule,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)

DESCRIPTION = """
Prédit si une petite molécule a une énergie de liaison **élevée** (`High_BFE`) ou **faible**
(`Low_BFE`) sur la protéine Sirtuin 6, à partir de 6 descripteurs moléculaires PaDEL.

Le modèle servi est la version du registre MLflow pointée par un alias, `production` par défaut
(posé à la main après revue). Chaque réponse indique la version qui l'a produite.
"""


def check_contract(model: ServedModel) -> None:
    """Refuse un modèle que cette version de l'API ne sait pas servir correctement."""
    if set(model.features) != set(Molecule.features()):
        raise RuntimeError(
            f"{model.name} v{model.version} attend {model.features}, "
            f"le schéma Molecule expose {Molecule.features()}"
        )
    if not hasattr(model.pipeline, "predict_proba"):
        raise RuntimeError(f"{model.name} v{model.version} ne fournit pas de probabilités")


async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Renvoie le 422 sans l'écho des valeurs reçues.

    NaN et infini ne sont pas sérialisables en JSON et feraient échouer la réponse d'erreur
    elle-même (500 au lieu de 422).
    """
    errors = [
        {key: value for key, value in error.items() if key != "input"} for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


def get_model(request: Request) -> ServedModel:
    return request.app.state.model


ServedModelDep = Annotated[ServedModel, Depends(get_model)]


def model_ref(model: ServedModel) -> ModelRef:
    return ModelRef(name=model.name, version=model.version, alias=model.alias)


router = APIRouter()


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@router.get("/health", tags=["monitoring"])
def health(model: ServedModelDep) -> Health:
    """Sonde de disponibilité : l'API ne répond qu'une fois le modèle chargé et validé."""
    return Health(status="ok", model=model_ref(model))


@router.get("/v1/model", tags=["modèle"])
def model_info(model: ServedModelDep) -> ModelInfo:
    """Fiche du modèle servi : provenance MLflow, descripteurs, hyperparamètres, métriques."""
    return ModelInfo(
        name=model.name,
        version=model.version,
        alias=model.alias,
        run_id=model.run_id,
        features=model.features,
        classes=CLASSES,
        positive_class=POSITIVE_CLASS,
        params=model.params,
        metrics=model.metrics,
    )


@router.post("/v1/predict", tags=["prédiction"])
def predict(molecule: Molecule, model: ServedModelDep) -> PredictionResponse:
    """Prédit la classe d'énergie de liaison d'une molécule."""
    prediction = model.predict([molecule.model_dump(by_alias=True)])[0]
    return PredictionResponse(**prediction, model=model_ref(model))


@router.post("/v1/predict/batch", tags=["prédiction"])
def predict_batch(batch: BatchRequest, model: ServedModelDep) -> BatchResponse:
    """Prédit plusieurs molécules en un appel ; les réponses suivent l'ordre de la requête."""
    rows = [molecule.model_dump(by_alias=True) for molecule in batch.molecules]
    return BatchResponse(predictions=model.predict(rows), model=model_ref(model))


def create_app(load_model: Callable[[], ServedModel] = load_from_registry) -> FastAPI:
    """Crée l'application FastAPI ; les tests remplacent `load_model` pour se passer de MLflow."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = load_model()
        check_contract(model)
        app.state.model = model
        yield

    app = FastAPI(
        title="Sirtuin6 BFE API",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        exception_handlers={RequestValidationError: validation_error},
    )
    # Origines navigateur autorisées à appeler l'API (`CORS_ORIGINS`, séparées par des virgules)
    origins = os.environ.get("CORS_ORIGINS", FRONTEND_URL).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    # Développement local (bouton play, `make run`) : rechargement à chaud du code de l'API
    uvicorn.run(
        "src.deploying.app.main:app",
        host="127.0.0.1",
        port=API_PORT,
        reload=True,
        reload_dirs=[str(DEPLOYING_PATH)],
    )
