"""Pipeline MLOps du modèle SIRTUIN6 : données -> modèles -> gate qualité -> alias staging.

Principes :
- la logique métier vit dans src/, le DAG ne fait qu'orchestrer ;
- un contrôle qualité bloque chaque étape sensible (données, performances) ;
- le registre MLflow est la source de vérité : on n'enregistre que le meilleur
  modèle, et seulement s'il passe le gate ;
- l'alias `production` n'est jamais posé ici : la promotion reste une décision
  humaine (revue du modèle staging dans MLflow).
"""

import pendulum
from airflow.sdk import dag, task

from config.ml_params import MODEL_NAMES

DEFAULT_ARGS = {"owner": "mlops", "retries": 1, "retry_delay": pendulum.duration(minutes=2)}


@dag(
    dag_id="sirtuin6_staging_model",
    description="Entraîne, valide et place le meilleur modèle en staging dans MLflow",
    schedule="0 3 * * 1",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "sirtuin6", "staging"],
)
def sirtuin6_staging_model():
    @task
    def extract_and_validate() -> int:
        """Extraction, typage et contrôles qualité des données."""
        from config.setting import CLEAN_FILE_PATH
        from src.pipeline.extract_data import extract_data, validate_data

        df = extract_data()
        validate_data(df)
        CLEAN_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_FILE_PATH, index=False)
        return len(df)

    @task
    def process(name: str) -> str:
        from src.pipeline.extract_data import load_clean_data
        from src.pipeline.process_data import (
            process_elastic,
            process_svm,
            process_trees,
        )

        {"elastic": process_elastic, "svm": process_svm, "trees": process_trees}[name](
            load_clean_data()
        )
        return name

    @task
    def optimize(name: str) -> dict:
        from src.lab.hyperparameters_optimizations import (
            optimize_elastic,
            optimize_svm,
            optimize_trees,
        )

        return {"elastic": optimize_elastic, "svm": optimize_svm, "trees": optimize_trees}[
            name
        ]()

    @task
    def build(name: str) -> str:
        from src.pipeline.build_models import build_elastic, build_svm, build_trees

        {"elastic": build_elastic, "svm": build_svm, "trees": build_trees}[name]()
        return name

    @task
    def benchmark() -> str:
        """Compare les modèles sur le jeu de test et retourne le meilleur."""
        from src.lab.benchmarking import main as run_benchmark
        from src.pipeline.train_final import best_model_name

        run_benchmark()
        return best_model_name()

    @task.short_circuit
    def quality_gate(best: str) -> bool:
        """Bloque la suite si les performances sont sous le seuil."""
        from src.monitoring.mlflow_staging import passes_quality_gate

        return passes_quality_gate(best)

    @task
    def register(best: str) -> str:
        """Entraîne sur toutes les données, logue et enregistre une nouvelle version."""
        from src.monitoring.mlflow_tracking import log_model
        from src.pipeline.extract_data import load_clean_data

        return str(log_model(best, load_clean_data(), promote=False))

    @task
    def set_staging_alias(best: str, version: str) -> str:
        from src.monitoring.mlflow_staging import stage_elastic, stage_svm, stage_trees

        {"elastic": stage_elastic, "svm": stage_svm, "trees": stage_trees}[best](version)
        return version

    @task
    def smoke_test(best: str, _staged: str) -> None:
        """Charge le modèle via l'alias staging, exactement comme un consommateur."""
        from src.monitoring.mlflow_staging import smoke_test as run_smoke_test

        run_smoke_test(best)

    data_ready = extract_and_validate()
    processed = process.expand(name=list(MODEL_NAMES))
    tuned = optimize.expand(name=list(MODEL_NAMES))
    built = build.expand(name=list(MODEL_NAMES))
    data_ready >> processed >> tuned >> built

    best = benchmark()
    built >> best

    gate = quality_gate(best)
    version = register(best)
    gate >> version

    staged = set_staging_alias(best, version)
    smoke_test(best, staged)


sirtuin6_staging_model()
