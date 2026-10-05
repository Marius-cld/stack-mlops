import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.ml_params import POSITIVE_CLASS, TARGET
from src.deploying.app.main import create_app
from src.deploying.app.model import ServedModel
from src.pipeline.extract_data import extract_data


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    """Données brutes versionnées : disponibles en CI sans lancer le pipeline."""
    return extract_data()


@pytest.fixture(scope="session")
def served_model(dataset: pd.DataFrame) -> ServedModel:
    """Pipeline sklearn léger à la place du modèle du registre : pas besoin de MLflow."""
    X = dataset.drop(columns=TARGET)
    y = (dataset[TARGET] == POSITIVE_CLASS).astype(int)
    return ServedModel(
        pipeline=make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y),
        name="sirtuin6-elastic",
        version="1",
        alias="test",
        run_id="test-run",
    )


@pytest.fixture
def client(served_model: ServedModel):
    with TestClient(create_app(load_model=lambda: served_model)) as client:
        yield client


@pytest.fixture
def molecule(dataset: pd.DataFrame) -> dict[str, float]:
    """Première molécule du dataset, au format JSON attendu par l'API."""
    return dataset.drop(columns=TARGET).iloc[0].to_dict()
