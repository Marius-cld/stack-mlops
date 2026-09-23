import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from config.ml_params import (
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_PREFIX,
    MLFLOW_PRODUCTION_ALIAS,
    POSITIVE_CLASS,
    TARGET,
)
from config.setting import BENCHMARK_PATH, MLFLOW_TRACKING_URI
from src.lab.hyperparameters_optimizations import load_best_params
from src.pipeline.extract_data import load_clean_data
from src.pipeline.train_final import best_model_name, make_pipeline


def registered_name(name: str) -> str:
    return f"{MLFLOW_MODEL_PREFIX}-{name}"


def setup() -> MlflowClient:
    """Pointe sur le serveur MLflow (surchargeable via MLFLOW_TRACKING_URI)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return MlflowClient()


def log_model(name: str, df: pd.DataFrame, promote: bool = False) -> str:
    """Entraîne le pipeline final, le logue et l'enregistre : une nouvelle version par appel."""
    client = setup()
    X = df.drop(columns=TARGET)
    y = (df[TARGET] == POSITIVE_CLASS).astype(int)
    pipeline = make_pipeline(name, X.columns).fit(X, y)

    benchmark = pd.read_csv(BENCHMARK_PATH, index_col="model").loc[name]
    with mlflow.start_run(run_name=f"final-{name}"):
        mlflow.set_tag("model", name)
        mlflow.log_params(load_best_params(name)["params"])
        mlflow.log_metric("cv_score", load_best_params(name)["cv_score"])
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in benchmark.items()})
        mlflow.log_input(mlflow.data.from_pandas(df, name="sirtuin6_clean"), context="training")
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X, pipeline.predict(X)),
            registered_model_name=registered_name(name),
        )

    version = info.registered_model_version
    if promote:
        client.set_registered_model_alias(
            registered_name(name), MLFLOW_PRODUCTION_ALIAS, version
        )
    return version


def load_production(name: str):
    """Charge la version alias 'production' d'un modèle enregistré."""
    setup()
    return mlflow.sklearn.load_model(
        f"models:/{registered_name(name)}@{MLFLOW_PRODUCTION_ALIAS}"
    )


def promote(name: str, version: int | str) -> None:
    setup().set_registered_model_alias(
        registered_name(name), MLFLOW_PRODUCTION_ALIAS, str(version)
    )


def main() -> None:
    best = best_model_name()
    df = load_clean_data()
    for name in ("elastic", "svm", "trees"):
        version = log_model(name, df, promote=(name == best))
        flag = f" -> alias {MLFLOW_PRODUCTION_ALIAS}" if name == best else ""
        print(f"{registered_name(name)} v{version}{flag}")


if __name__ == "__main__":
    main()
