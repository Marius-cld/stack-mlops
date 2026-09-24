from pathlib import Path

# Racine du projet (config/ est à un niveau sous la racine).
# Calculée depuis ce fichier : indépendante du dossier courant, identique en local
# (bouton play, python -m) et dans Airflow (/opt/airflow/project).
# Tous les chemins du projet sont définis ici, jamais dans les scripts.
ROOT_PATH = Path(__file__).resolve().parent.parent

# Services
SERVICES_PATH = ROOT_PATH / "services"
AIRFLOW_PATH = SERVICES_PATH / "airflow"
AIRFLOW_DAG_PATH = AIRFLOW_PATH / "dag"
MLFLOW_PATH = SERVICES_PATH / "mlflow"
MLFLOW_DATA_PATH = MLFLOW_PATH / "data"
MLFLOW_PORT = 5001  # 5000 est pris par AirPlay sur macOS
MLFLOW_TRACKING_URI = f"http://localhost:{MLFLOW_PORT}"
EVIDENTLY_PATH = SERVICES_PATH / "evidentlyia"
EVIDENTLY_WORKSPACE_PATH = EVIDENTLY_PATH / "workspace"
EVIDENTLY_PORT = 8000
EVIDENTLY_WORKSPACE_URL = f"http://localhost:{EVIDENTLY_PORT}"

# Data
DATA_PATH = ROOT_PATH / "data"
RAW_PATH = DATA_PATH / "raw"
PROCESS_PATH = DATA_PATH / "processed"
FIGURES_PATH = DATA_PATH / "figures"
MODELS_PATH = DATA_PATH / "models"
RAW_FILE_PATH = RAW_PATH / "SIRTUIN6.csv"
CLEAN_FILE_PATH = PROCESS_PATH / "sirtuin6_clean.csv"

# Artifacts (données traitées, préprocesseurs, hyperparamètres, benchmark)
ARTIFACTS_PATH = DATA_PATH / "artifacts"
BENCHMARK_PATH = ARTIFACTS_PATH / "benchmark.csv"

# Source
SRC_PATH = ROOT_PATH / "src"
DESCRIPTIVE_STATISTICS_PATH = SRC_PATH / "descriptive_statistics"
LAB_PATH = SRC_PATH / "lab"
PIPELINE_PATH = SRC_PATH / "pipeline"
ROBUSTNESS_PATH = SRC_PATH / "robustness"
MONITORING_PATH = SRC_PATH / "monitoring"


def figure_path(name: str) -> Path:
    return FIGURES_PATH / f"{name}.png"


# Chemins par modèle (name = "elastic" | "svm" | "trees")
def artifacts_dir(name: str) -> Path:
    return ARTIFACTS_PATH / name


def split_path(name: str, split: str) -> Path:
    """split = "X_train" | "X_test" | "y_train" | "y_test"."""
    return artifacts_dir(name) / f"{split}.csv"


def preprocessor_path(name: str) -> Path:
    return artifacts_dir(name) / "preprocessor.joblib"


def learned_stats_path(name: str) -> Path:
    return artifacts_dir(name) / "learned_stats.json"


def best_params_path(name: str) -> Path:
    return artifacts_dir(name) / "best_params.json"


def model_path(name: str) -> Path:
    return MODELS_PATH / f"{name}.joblib"


def final_model_path(name: str) -> Path:
    return MODELS_PATH / f"final_{name}.joblib"


# Robustesse
ROBUSTNESS_ARTIFACTS_PATH = ARTIFACTS_PATH / "robustness"


def robustness_result_path(test: str) -> Path:
    return ROBUSTNESS_ARTIFACTS_PATH / f"{test}.csv"
