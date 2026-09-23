import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from config.ml_params import (
    NOISE_CV_REPEATS,
    NOISE_LEVELS,
    RANDOM_STATE,
    ROBUSTNESS_CV_SPLITS,
)
from src.pipeline.train_final import make_pipeline
from src.robustness.common import load_xy, positive_score, save_results


def _noise(name: str) -> pd.DataFrame:
    """Bruit gaussien sur les variables du fold de test.

    L'écart-type du bruit vaut `niveau` × écart-type de la variable (train).
    """
    X, y = load_xy()
    pipeline = make_pipeline(name, X.columns)
    cv = RepeatedStratifiedKFold(
        n_splits=ROBUSTNESS_CV_SPLITS,
        n_repeats=NOISE_CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []
    for train, test in cv.split(X, y):
        model = clone(pipeline).fit(X.iloc[train], y.iloc[train])
        std = X.iloc[train].std()
        for level in NOISE_LEVELS:
            X_noisy = X.iloc[test] + rng.normal(size=X.iloc[test].shape) * std.to_numpy() * level
            rows.append(
                {
                    "noise_level": level,
                    "roc_auc": roc_auc_score(
                        y.iloc[test], positive_score(model, X_noisy)
                    ),
                    "accuracy": accuracy_score(y.iloc[test], model.predict(X_noisy)),
                }
            )

    table = (
        pd.DataFrame(rows)
        .groupby("noise_level")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .reset_index()
    )
    table.insert(0, "model", name)
    return table


def noise_elastic() -> pd.DataFrame:
    return _noise("elastic")


def noise_svm() -> pd.DataFrame:
    return _noise("svm")


def noise_trees() -> pd.DataFrame:
    return _noise("trees")


def main() -> pd.DataFrame:
    return save_results("gaussian_noise", [noise_elastic(), noise_svm(), noise_trees()])


if __name__ == "__main__":
    main()
