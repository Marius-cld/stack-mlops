import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from config.ml_params import (
    CV_REPEATS,
    CV_SPLITS,
    ELASTIC_GRID,
    RANDOM_STATE,
    SCORING,
    SVM_GRID,
    TREES_GRID,
)
from src.lab.hyperparameters_optimizations import load_best_params
from src.pipeline.train_final import make_pipeline
from src.robustness.common import load_xy, save_results


def _sensitivity(name: str, grid: dict) -> pd.DataFrame:
    """Fait varier un hyperparamètre à la fois autour de l'optimum."""
    X, y = load_xy()
    best = load_best_params(name)["params"]
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )

    rows = []
    for param, values in grid.items():
        for value in values:
            pipeline = make_pipeline(name, X.columns).set_params(
                **{f"model__{key}": val for key, val in {**best, param: value}.items()}
            )
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            rows.append(
                {
                    "model": name,
                    "param": param,
                    "value": str(value),
                    "is_best": value == best[param],
                    f"{SCORING}_mean": scores.mean(),
                    f"{SCORING}_std": scores.std(),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_elastic() -> pd.DataFrame:
    return _sensitivity("elastic", ELASTIC_GRID)


def sensitivity_svm() -> pd.DataFrame:
    return _sensitivity("svm", SVM_GRID)


def sensitivity_trees() -> pd.DataFrame:
    return _sensitivity("trees", TREES_GRID)


def main() -> pd.DataFrame:
    return save_results(
        "hyperparameters_sensitivity",
        [sensitivity_elastic(), sensitivity_svm(), sensitivity_trees()],
    )


if __name__ == "__main__":
    main()
