"""Modèle servi : chargé une fois depuis le registre MLflow, partagé par toutes les requêtes."""

import logging
import os
import urllib.request
from dataclasses import dataclass, field

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from config.ml_params import (
    CLASSES,
    MLFLOW_MODEL_PREFIX,
    POSITIVE_CLASS,
    SERVING_ALIAS,
    SERVING_MODEL,
)
from config.settings import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

# Encodage de la cible à l'entraînement (`process_data.split`) :
# 1 = `POSITIVE_CLASS`, 0 = l'autre classe
LABELS = {int(label == POSITIVE_CLASS): label for label in CLASSES}


@dataclass(frozen=True)
class ServedModel:
    """Pipeline ajusté (préprocesseur + estimateur) et sa provenance dans le registre."""

    pipeline: Pipeline
    name: str
    version: str
    alias: str
    run_id: str
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        """Descripteurs attendus, dans l'ordre vu à l'entraînement."""
        return list(self.pipeline.feature_names_in_)

    def predict(self, rows: list[dict[str, float]]) -> list[dict]:
        """Prédit classe et probabilités, une entrée par ligne et dans le même ordre."""
        X = pd.DataFrame(rows, columns=self.features)
        labels = [LABELS[int(cls)] for cls in self.pipeline.classes_]
        return [
            {"label": labels[proba.argmax()], "probabilities": dict(zip(labels, proba.tolist()))}
            for proba in self.pipeline.predict_proba(X)
        ]


def tracking_uri() -> str:
    """Retourne l'URI du serveur MLflow (surchargeable via `MLFLOW_TRACKING_URI`)."""
    return os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)


def registry_reachable(timeout: float = 5) -> bool:
    """Vérifie que le serveur MLflow répond, sans les ~4 min de tentatives du client MLflow."""
    try:
        urllib.request.urlopen(f"{tracking_uri()}/health", timeout=timeout)
    except OSError:
        return False
    return True


def load_from_registry() -> ServedModel:
    """Charge la version pointée par l'alias : `production` par défaut, ou `MODEL_ALIAS`."""
    if not registry_reachable():
        raise RuntimeError(f"Registre MLflow injoignable : {tracking_uri()}")
    mlflow.set_tracking_uri(tracking_uri())
    name = f"{MLFLOW_MODEL_PREFIX}-{SERVING_MODEL}"
    alias = os.environ.get("MODEL_ALIAS", SERVING_ALIAS)

    client = MlflowClient()
    # Alias résolu une seule fois, puis chargement par numéro de version : la version annoncée
    # est exactement celle chargée, même si l'alias est déplacé pendant le chargement
    version = client.get_model_version_by_alias(name, alias)
    pipeline = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
    run = client.get_run(version.run_id)
    logger.info(
        "Modèle chargé : %s v%s (alias %s, run %s)", name, version.version, alias, version.run_id
    )
    return ServedModel(
        pipeline=pipeline,
        name=name,
        version=str(version.version),
        alias=alias,
        run_id=version.run_id,
        params=run.data.params,
        metrics=run.data.metrics,
    )
