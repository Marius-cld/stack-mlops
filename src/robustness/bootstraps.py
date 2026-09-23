import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score

from config.ml_params import N_BOOTSTRAPS, RANDOM_STATE
from src.pipeline.train_final import make_pipeline
from src.robustness.common import load_xy, positive_score, save_results, summarize


def _bootstrap(name: str) -> pd.DataFrame:
    """Entraîne sur un échantillon bootstrap, évalue sur les lignes out-of-bag."""
    X, y = load_xy()
    pipeline = make_pipeline(name, X.columns)
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)

    rows = []
    for _ in range(N_BOOTSTRAPS):
        train = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), train)
        if y.iloc[train].nunique() < 2 or y.iloc[oob].nunique() < 2:
            continue
        model = clone(pipeline).fit(X.iloc[train], y.iloc[train])
        rows.append(
            {
                "roc_auc": roc_auc_score(
                    y.iloc[oob], positive_score(model, X.iloc[oob])
                ),
                "accuracy": accuracy_score(y.iloc[oob], model.predict(X.iloc[oob])),
            }
        )

    table = summarize(pd.DataFrame(rows)).rename_axis("metric").reset_index()
    table.insert(0, "model", name)
    return table


def bootstrap_elastic() -> pd.DataFrame:
    return _bootstrap("elastic")


def bootstrap_svm() -> pd.DataFrame:
    return _bootstrap("svm")


def bootstrap_trees() -> pd.DataFrame:
    return _bootstrap("trees")


def main() -> pd.DataFrame:
    return save_results(
        "bootstrap", [bootstrap_elastic(), bootstrap_svm(), bootstrap_trees()]
    )


if __name__ == "__main__":
    main()
