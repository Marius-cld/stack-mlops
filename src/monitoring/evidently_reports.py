import os

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import RemoteWorkspace
from sklearn.model_selection import train_test_split

from config.ml_params import RANDOM_STATE, TARGET
from config.settings import CLEAN_FILE_PATH, EVIDENTLY_WORKSPACE_URL, RAW_FILE_PATH
from src.pipeline.process_data import split

PROJECT_NAME = "sirtuin6"


def get_workspace() -> RemoteWorkspace:
    """Pointe sur le serveur Evidently (surchargeable via EVIDENTLY_WORKSPACE_URL)."""
    return RemoteWorkspace(os.environ.get("EVIDENTLY_WORKSPACE_URL", EVIDENTLY_WORKSPACE_URL))


def load_data() -> pd.DataFrame:
    path = CLEAN_FILE_PATH if CLEAN_FILE_PATH.exists() else RAW_FILE_PATH
    return pd.read_csv(path)


def to_dataset(df: pd.DataFrame) -> Dataset:
    numerical = [c for c in df.columns if c != TARGET]
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=[TARGET])
    return Dataset.from_pandas(df, data_definition=definition)


def add_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> str:
    """Data summary + data drift (current vs reference), ajouté au workspace. Retourne l'id du snapshot."""
    report = Report([DataSummaryPreset(), DataDriftPreset()])
    snapshot = report.run(
        to_dataset(current), to_dataset(reference), tags=tags, metadata=metadata
    )

    workspace = get_workspace()
    project = next(iter(workspace.search_project(PROJECT_NAME)), None)
    if project is None:
        project = workspace.create_project(PROJECT_NAME)
        project.description = "Data summary et data drift du dataset SIRTUIN6"
        project.save()
    ref = workspace.add_run(project.id, snapshot)
    print(f"Rapport ajouté au workspace Evidently {workspace.get_url()}")
    return str(ref.id)


def report_split(name: str, version: str | None = None) -> str:
    """Référence = train, courant = test : le split exact utilisé pour entraîner et évaluer le modèle."""
    df = load_data()
    X_train, X_test, _, _ = split(df)
    tags = [name, "train_vs_test"]
    metadata = {"model": name}
    if version is not None:
        tags.append(f"v{version}")
        metadata["mlflow_version"] = str(version)
    return add_report(df.loc[X_train.index], df.loc[X_test.index], tags, metadata)


def report_elastic(version: str | None = None) -> str:
    return report_split("elastic", version)


def main() -> None:
    df = load_data()
    reference, current = train_test_split(
        df, test_size=0.3, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    add_report(reference, current)


if __name__ == "__main__":
    main()
