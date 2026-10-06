"""Tests de l'API de serving : contrat d'entrée et de sortie, démarrage, registre MLflow."""

import json
from dataclasses import replace
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config.ml_params import CLASSES, MAX_BATCH_SIZE, POSITIVE_CLASS, TARGET
from config.settings import OPENAPI_PATH
from src.deploying.app.main import create_app
from src.deploying.app.model import load_from_registry, registry_reachable
from src.deploying.app.schemas import Label, Molecule


def test_health(client):
    """La sonde répond une fois le modèle chargé et indique la version servie."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": {"name": "sirtuin6-elastic", "version": "1", "alias": "test"},
    }


def test_model_info(client):
    """La fiche du modèle expose descripteurs, classes et provenance MLflow."""
    info = client.get("/v1/model").json()
    assert set(info["features"]) == set(Molecule.features())
    assert info["classes"] == CLASSES
    assert info["positive_class"] == POSITIVE_CLASS
    assert info["run_id"] == "test-run"


def test_predict(client, molecule):
    """La réponse donne une probabilité par classe (somme = 1) et le label le plus probable."""
    response = client.post("/v1/predict", json=molecule)
    assert response.status_code == 200
    body = response.json()
    probabilities = body["probabilities"]
    assert set(probabilities) == set(CLASSES)
    assert sum(probabilities.values()) == pytest.approx(1)
    assert body["label"] == max(probabilities, key=probabilities.get)
    assert body["model"]["version"] == "1"


def test_batch_matches_pipeline(client, served_model, dataset):
    """Aucun écart entre l'API et le pipeline appelé directement, ordre des molécules conservé."""
    X = dataset.drop(columns=TARGET)
    response = client.post("/v1/predict/batch", json={"molecules": X.to_dict(orient="records")})
    assert response.status_code == 200
    predictions = response.json()["predictions"]

    pipeline = served_model.pipeline
    expected_labels = [POSITIVE_CLASS if y == 1 else "Low_BFE" for y in pipeline.predict(X)]
    assert [p["label"] for p in predictions] == expected_labels
    assert [p["probabilities"][POSITIVE_CLASS] for p in predictions] == pytest.approx(
        pipeline.predict_proba(X)[:, 1].tolist()
    )


INVALID_MOLECULES = {
    "descripteur manquant": lambda m: {k: v for k, v in m.items() if k != "FMF"},
    "descripteur inconnu": lambda m: {**m, "logP": 1.0},
    "texte": lambda m: {**m, "SC-5": "abc"},
    "NaN": lambda m: {**m, "SC-5": float("nan")},
    "infini": lambda m: {**m, "SC-5": float("inf")},
}


@pytest.mark.parametrize("alter", INVALID_MOLECULES.values(), ids=list(INVALID_MOLECULES))
def test_invalid_molecule_rejected(client, molecule, alter):
    """Une molécule hors contrat est refusée en 422."""
    # `json.dumps` écrit NaN / Infinity, que le parseur JSON de l'API accepte :
    # c'est le schéma qui doit les refuser
    response = client.post(
        "/v1/predict",
        content=json.dumps(alter(molecule)),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("size", [0, MAX_BATCH_SIZE + 1])
def test_batch_size_limits(client, molecule, size):
    """Un lot vide ou au-delà de `MAX_BATCH_SIZE` est refusé en 422."""
    response = client.post("/v1/predict/batch", json={"molecules": [molecule] * size})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("dropped", "estimator", "error"),
    [(["FMF"], LogisticRegression(), "attend"), ([], SVC(), "probabilités")],
    ids=["descripteur manquant", "sans predict_proba"],
)
def test_unservable_model_prevents_startup(served_model, dataset, dropped, estimator, error):
    """Un modèle incompatible avec le contrat de l'API ne doit jamais être servi."""
    X = dataset.drop(columns=[TARGET, *dropped])
    y = (dataset[TARGET] == POSITIVE_CLASS).astype(int)
    model = replace(served_model, pipeline=make_pipeline(StandardScaler(), estimator).fit(X, y))
    with pytest.raises(RuntimeError, match=error):
        with TestClient(create_app(load_model=lambda: model)):
            pass


def test_label_type_matches_config():
    """Le type `Label`, écrit en clair, reste aligné sur `CLASSES`."""
    assert list(get_args(Label)) == CLASSES


def test_openapi_schema(client):
    """Le schéma OpenAPI (source des types TypeScript du frontend) se génère."""
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/v1/model", "/v1/predict", "/v1/predict/batch"} <= paths.keys()


def test_openapi_contract_up_to_date(client):
    """Le contrat publié pour le frontend (`openapi.json`) est celui que l'API sert réellement."""
    published = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    assert published == client.get("/openapi.json").json(), "contrat obsolète : `make openapi`"


def test_unreachable_registry_fails_fast(monkeypatch):
    """Registre absent : erreur immédiate et explicite, pas 4 min de tentatives du client MLflow."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    with pytest.raises(RuntimeError, match="injoignable"):
        load_from_registry()


@pytest.mark.integration
@pytest.mark.skipif(not registry_reachable(timeout=2), reason="serveur MLflow injoignable")
def test_registry_model_serves_predictions(molecule):
    """Le modèle réellement promu dans le registre respecte le contrat de l'API et prédit."""
    with TestClient(create_app()) as client:
        response = client.post("/v1/predict", json=molecule)
    assert response.status_code == 200
    assert response.json()["label"] in CLASSES
