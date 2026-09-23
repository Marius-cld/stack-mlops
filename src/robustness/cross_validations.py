import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from config.ml_params import (
    RANDOM_STATE,
    ROBUSTNESS_CV_REPEATS,
    ROBUSTNESS_CV_SPLITS,
    ROBUSTNESS_METRICS,
)
from src.pipeline.train_final import make_pipeline
from src.robustness.common import load_xy, save_results, summarize


def _cross_validate(name: str) -> pd.DataFrame:
    """Validation croisée stratifiée répétée sur toutes les données."""
    X, y = load_xy()
    cv = RepeatedStratifiedKFold(
        n_splits=ROBUSTNESS_CV_SPLITS,
        n_repeats=ROBUSTNESS_CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        make_pipeline(name, X.columns),
        X,
        y,
        cv=cv,
        scoring=list(ROBUSTNESS_METRICS),
        n_jobs=-1,
    )
    values = pd.DataFrame({m: scores[f"test_{m}"] for m in ROBUSTNESS_METRICS})
    table = summarize(values).rename_axis("metric").reset_index()
    table.insert(0, "model", name)
    return table


def cv_elastic() -> pd.DataFrame:
    return _cross_validate("elastic")


def cv_svm() -> pd.DataFrame:
    return _cross_validate("svm")


def cv_trees() -> pd.DataFrame:
    return _cross_validate("trees")


def main() -> pd.DataFrame:
    return save_results("cross_validation", [cv_elastic(), cv_svm(), cv_trees()])


if __name__ == "__main__":
    main()
