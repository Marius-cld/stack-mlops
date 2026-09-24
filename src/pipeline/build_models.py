import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from config.ml_params import ELASTIC_PARAMS, SVM_PARAMS, TREES_PARAMS
from config.settings import MODELS_PATH, model_path
from src.lab.hyperparameters_optimizations import load_best_params
from src.pipeline.process_data import load_processed


def _build(name: str, estimator):
    X_train, _, y_train, _ = load_processed(name)
    estimator.set_params(**load_best_params(name)["params"])
    estimator.fit(X_train, y_train)

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_path(name))
    return estimator


def build_elastic():
    return _build("elastic", LogisticRegression(**ELASTIC_PARAMS))


def build_svm():
    return _build("svm", SVC(**SVM_PARAMS))


def build_trees():
    return _build("trees", RandomForestClassifier(**TREES_PARAMS))


def main() -> None:
    build_elastic()
    build_svm()
    build_trees()


if __name__ == "__main__":
    main()
