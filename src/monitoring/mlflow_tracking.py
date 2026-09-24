import os
from importlib.metadata import version

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config.ml_params import (
    MLFLOW_CANDIDATE_ALIAS,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_PREFIX,
    MLFLOW_PRODUCTION_ALIAS,
    POSITIVE_CLASS,
    TARGET,
)
from config.settings import (
    BENCHMARK_PATH,
    MLFLOW_TRACKING_URI,
    final_model_path,
    model_path,
    preprocessor_path,
)
from src.lab.hyperparameters_optimizations import load_best_params
from src.pipeline.extract_data import load_clean_data
from src.pipeline.process_data import split
from src.pipeline.train_final import best_model_name, make_pipeline


def registered_name(name: str) -> str:
    return f"{MLFLOW_MODEL_PREFIX}-{name}"


def setup() -> MlflowClient:
    """Pointe sur le serveur MLflow (surchargeable via MLFLOW_TRACKING_URI)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    # Un experiment supprimé depuis l'UI reste en corbeille (soft delete) : on le restaure.
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return client


def _pip_requirements() -> list[str]:
    """Dépendances figées depuis l'environnement courant : évite le sous-processus
    d'inférence de log_model (il recharge le modèle et fait exploser la mémoire du worker)."""
    return [
        f"{pkg}=={version(pkg)}"
        for pkg in ("mlflow", "scikit-learn", "pandas", "numpy", "cloudpickle")
    ]


def _log_pipeline(
    name: str,
    pipeline: Pipeline,
    df: pd.DataFrame,
    dataset_name: str,
    run_name: str,
    metrics: dict,
    tags: dict | None = None,
) -> str:
    """Logue un pipeline ajusté (params, métriques, dataset, modèle) : une nouvelle version par appel."""
    setup()
    X = df.drop(columns=TARGET)
    best = load_best_params(name)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model": name, **(tags or {})})
        mlflow.log_params(best["params"])
        mlflow.log_metric("cv_score", best["cv_score"])
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in metrics.items()})
        mlflow.log_input(mlflow.data.from_pandas(df, name=dataset_name), context="training")
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X.head(100), pipeline.predict(X.head(100))),
            pip_requirements=_pip_requirements(),
            registered_model_name=registered_name(name),
        )
    return str(info.registered_model_version)


def log_model(name: str, df: pd.DataFrame, promote: bool = False) -> str:
    """Entraîne le pipeline final, le logue et l'enregistre : une nouvelle version par appel."""
    X = df.drop(columns=TARGET)
    y = (df[TARGET] == POSITIVE_CLASS).astype(int)
    pipeline = make_pipeline(name, X.columns).fit(X, y)

    benchmark = pd.read_csv(BENCHMARK_PATH, index_col="model").loc[name]
    version = _log_pipeline(
        name, pipeline, df, "sirtuin6_clean", f"final-{name}", benchmark.to_dict()
    )
    if promote:
        set_alias(name, MLFLOW_PRODUCTION_ALIAS, version)
    return version


def trained_pipeline(name: str, X_train: pd.DataFrame) -> Pipeline:
    """Pipeline évalué au benchmark : préprocesseur (process_data) et modèle (build_models) ajustés sur le train."""
    steps = []
    if preprocessor_path(name).exists():
        preprocessor = joblib.load(preprocessor_path(name))
        # process_data remet les colonnes dans l'ordre d'origine après transformation
        order = ColumnTransformer(
            [("order", "passthrough", list(X_train.columns))], verbose_feature_names_out=False
        ).set_output(transform="pandas")
        order.fit(preprocessor.transform(X_train))
        steps += [("preprocessor", preprocessor), ("order", order)]
    steps.append(("model", joblib.load(model_path(name))))
    return Pipeline(steps)


def log_candidate(name: str, metrics: dict) -> str:
    """Enregistre le modèle entraîné sur le train, avec ses métriques de test, alias candidate."""
    df = load_clean_data()
    X_train, _, _, _ = split(df)
    train = df.loc[X_train.index]
    version = _log_pipeline(
        name,
        trained_pipeline(name, X_train),
        train,
        "sirtuin6_train",
        f"candidate-{name}",
        metrics,
        tags={"training_data": "train_split"},
    )
    set_alias(name, MLFLOW_CANDIDATE_ALIAS, version)
    return version


def log_final(name: str, metrics: dict, candidate_version: str, tags: dict | None = None) -> str:
    """Enregistre le modèle réentraîné sur tout le dataset (train_final), lié à sa version candidate."""
    return _log_pipeline(
        name,
        joblib.load(final_model_path(name)),
        load_clean_data(),
        "sirtuin6_clean",
        f"final-{name}",
        metrics,
        tags={
            "training_data": "full_dataset",
            "candidate_version": str(candidate_version),
            **(tags or {}),
        },
    )


def log_candidate_elastic(metrics: dict) -> str:
    return log_candidate("elastic", metrics)


def log_final_elastic(metrics: dict, candidate_version: str, tags: dict | None = None) -> str:
    return log_final("elastic", metrics, candidate_version, tags)


def set_alias(name: str, alias: str, version: int | str) -> None:
    setup().set_registered_model_alias(registered_name(name), alias, str(version))


def load_production(name: str):
    """Charge la version alias 'production' d'un modèle enregistré."""
    setup()
    return mlflow.sklearn.load_model(
        f"models:/{registered_name(name)}@{MLFLOW_PRODUCTION_ALIAS}"
    )


def promote(name: str, version: int | str) -> None:
    set_alias(name, MLFLOW_PRODUCTION_ALIAS, version)


def main() -> None:
    best = best_model_name()
    df = load_clean_data()
    for name in ("elastic", "svm", "trees"):
        version = log_model(name, df, promote=(name == best))
        flag = f" -> alias {MLFLOW_PRODUCTION_ALIAS}" if name == best else ""
        print(f"{registered_name(name)} v{version}{flag}")


if __name__ == "__main__":
    main()
