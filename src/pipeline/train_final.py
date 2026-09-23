import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config.ml_params import (
    ELASTIC_PARAMS,
    POSITIVE_CLASS,
    SVM_PARAMS,
    TARGET,
    TREES_PARAMS,
)
from config.setting import BENCHMARK_PATH, MODELS_PATH, final_model_path
from src.lab.hyperparameters_optimizations import load_best_params
from src.pipeline.extract_data import load_clean_data
from src.pipeline.process_data import make_preprocessor

BEST_METRIC = "roc_auc"


ESTIMATORS = {
    "elastic": (LogisticRegression, ELASTIC_PARAMS),
    "svm": (SVC, SVM_PARAMS),
    "trees": (RandomForestClassifier, TREES_PARAMS),
}


def make_pipeline(name: str, columns) -> Pipeline:
    """Pipeline non ajusté (préprocesseur + modèle) avec les meilleurs hyperparamètres."""
    estimator_class, fixed = ESTIMATORS[name]
    estimator = estimator_class(**fixed).set_params(**load_best_params(name)["params"])
    steps = []
    preprocessor = make_preprocessor(name, columns)
    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _train_final(name: str, df: pd.DataFrame) -> Pipeline:
    """Entraîne préprocesseur + modèle sur l'ensemble des données."""
    X = df.drop(columns=TARGET)
    y = (df[TARGET] == POSITIVE_CLASS).astype(int)

    pipeline = make_pipeline(name, X.columns).fit(X, y)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, final_model_path(name))
    return pipeline


def train_final_elastic(df: pd.DataFrame) -> Pipeline:
    return _train_final("elastic", df)


def train_final_svm(df: pd.DataFrame) -> Pipeline:
    return _train_final("svm", df)


def train_final_trees(df: pd.DataFrame) -> Pipeline:
    return _train_final("trees", df)


TRAINERS = {
    "elastic": train_final_elastic,
    "svm": train_final_svm,
    "trees": train_final_trees,
}


def best_model_name() -> str:
    benchmark = pd.read_csv(BENCHMARK_PATH, index_col="model")
    return benchmark[BEST_METRIC].idxmax()


def main() -> None:
    name = best_model_name()
    print(f"Meilleur modèle ({BEST_METRIC}) : {name}")
    TRAINERS[name](load_clean_data())
    print(f"-> {final_model_path(name)}")


if __name__ == "__main__":
    main()
