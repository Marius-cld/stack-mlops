import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC

from config.ml_params import (
    CV_REPEATS,
    CV_SPLITS,
    ELASTIC_GRID,
    ELASTIC_PARAMS,
    RANDOM_STATE,
    SCORING,
    SVM_GRID,
    SVM_PARAMS,
    TREES_GRID,
    TREES_PARAMS,
)
from config.setting import best_params_path
from src.pipeline.process_data import load_processed


def _cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)


def load_best_params(name: str) -> dict:
    return json.loads(best_params_path(name).read_text())


def _optimize(name: str, estimator, grid: dict) -> dict:
    X_train, _, y_train, _ = load_processed(name)
    search = GridSearchCV(estimator, grid, scoring=SCORING, cv=_cv(), n_jobs=-1)
    search.fit(X_train, y_train)

    params = {key: _native(value) for key, value in search.best_params_.items()}
    best_params_path(name).write_text(
        json.dumps({"params": params, "cv_score": search.best_score_}, indent=2)
    )
    print(f"{name}: {params} ({SCORING}={search.best_score_:.3f})")
    return params


def _native(value):
    return value.item() if hasattr(value, "item") else value


def optimize_elastic() -> dict:
    return _optimize("elastic", LogisticRegression(**ELASTIC_PARAMS), ELASTIC_GRID)


def optimize_svm() -> dict:
    return _optimize("svm", SVC(**SVM_PARAMS), SVM_GRID)


def optimize_trees() -> dict:
    return _optimize("trees", RandomForestClassifier(**TREES_PARAMS), TREES_GRID)


def main() -> None:
    optimize_elastic()
    optimize_svm()
    optimize_trees()


if __name__ == "__main__":
    main()
