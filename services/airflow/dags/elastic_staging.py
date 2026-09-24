"""Pipeline MLOps du champion elastic (choisi par les scripts du lab : benchmark, robustesse).

extraction -> traitement (split) -> optimisation -> construction -> évaluation test
-> gate qualité -> MLflow (version + alias candidate) -> Evidently (train vs test)
-> réentraînement sur tout le dataset -> MLflow (version + alias staging) -> smoke test

Principes :
- la logique métier vit dans src/, le DAG ne fait qu'orchestrer ;
- le modèle évalué sur le test est versionné tel quel (alias `candidate`) : ses
  métriques sont honnêtes ;
- le modèle servi est réentraîné sur tout le dataset avec les mêmes hyperparamètres,
  versionné et relié à sa version candidate (tag `candidate_version`) ;
- l'alias `production` n'est jamais posé ici : la promotion reste une décision humaine.
"""

import pendulum
from airflow.sdk import dag, task

MODEL = "elastic"
DEFAULT_ARGS = {"owner": "mlops", "retries": 1, "retry_delay": pendulum.duration(minutes=2)}


@dag(
    dag_id="sirtuin6_elastic_staging",
    description="Entraîne, versionne et place le champion elastic en staging dans MLflow",
    schedule="0 3 * * 1",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "sirtuin6", "elastic", "staging"],
)
def sirtuin6_elastic_staging():
    @task
    def extract_and_validate() -> int:
        """Extraction, typage et contrôles qualité des données."""
        from config.settings import CLEAN_FILE_PATH
        from src.pipeline.extract_data import extract_data, validate_data

        df = extract_data()
        validate_data(df)
        CLEAN_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_FILE_PATH, index=False)
        return len(df)

    @task
    def process() -> None:
        """Split train/test stratifié + Yeo-Johnson et standardisation ajustés sur le train."""
        from src.pipeline.extract_data import load_clean_data
        from src.pipeline.process_data import process_elastic

        process_elastic(load_clean_data())

    @task
    def optimize() -> dict:
        """GridSearch en CV répétée sur le train uniquement."""
        from src.lab.hyperparameters_optimizations import optimize_elastic

        return optimize_elastic()

    @task
    def build() -> None:
        """Ajuste le modèle sur le train avec les meilleurs hyperparamètres."""
        from src.pipeline.build_models import build_elastic

        build_elastic()

    @task
    def evaluate() -> dict:
        """Métriques sur le jeu de test (jamais vu à l'entraînement ni à l'optimisation)."""
        from src.lab.benchmarking import benchmark_elastic

        metrics = benchmark_elastic()
        metrics.pop("model")
        return {k: float(v) for k, v in metrics.items()}

    @task.short_circuit
    def quality_gate(metrics: dict) -> bool:
        """Bloque la suite si l'AUC test ou l'AUC moyen en CV répétée est sous le seuil."""
        from src.monitoring.mlflow_staging import passes_quality_gate

        return passes_quality_gate(MODEL, test_auc=metrics["roc_auc"])

    @task
    def register_candidate(metrics: dict) -> str:
        """Versionne le modèle entraîné sur le train (run historisé + alias candidate)."""
        from src.monitoring.mlflow_tracking import log_candidate_elastic

        return log_candidate_elastic(metrics)

    @task
    def evidently_report(version: str) -> str:
        """Data summary + drift train vs test dans le workspace Evidently, tagué avec la version."""
        from src.monitoring.evidently_reports import report_elastic

        return report_elastic(version)

    @task
    def train_final() -> None:
        """Réentraîne préprocesseur + modèle sur l'intégralité du dataset."""
        from src.pipeline.extract_data import load_clean_data
        from src.pipeline.train_final import train_final_elastic

        train_final_elastic(load_clean_data())

    @task
    def register_final(metrics: dict, candidate_version: str, snapshot_id: str) -> str:
        """Versionne le modèle final, relié à sa version candidate et au rapport Evidently."""
        from src.monitoring.mlflow_tracking import log_final_elastic

        return log_final_elastic(
            metrics, candidate_version, tags={"evidently_snapshot": snapshot_id}
        )

    @task
    def set_staging_alias(version: str) -> str:
        from src.monitoring.mlflow_staging import stage_elastic

        return stage_elastic(version)

    @task
    def smoke_test(_staged: str) -> None:
        """Charge le modèle via l'alias staging, exactement comme un consommateur."""
        from src.monitoring.mlflow_staging import smoke_test as run_smoke_test

        run_smoke_test(MODEL)

    data_ready = extract_and_validate()
    processed = process()
    tuned = optimize()
    built = build()
    metrics = evaluate()
    data_ready >> processed >> tuned >> built >> metrics

    gate = quality_gate(metrics)
    candidate = register_candidate(metrics)
    gate >> candidate

    snapshot = evidently_report(candidate)
    final = train_final()
    snapshot >> final

    staged = set_staging_alias(register_final(metrics, candidate, snapshot))
    smoke_test(staged)


sirtuin6_elastic_staging()
