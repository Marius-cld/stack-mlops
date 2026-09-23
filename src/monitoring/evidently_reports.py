import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import Workspace
from sklearn.model_selection import train_test_split

from config.setting import CLEAN_FILE_PATH, EVIDENTLY_WORKSPACE_PATH, RAW_FILE_PATH

PROJECT_NAME = "sirtuin6"
TARGET = "Class"


def load_data() -> pd.DataFrame:
    path = CLEAN_FILE_PATH if CLEAN_FILE_PATH.exists() else RAW_FILE_PATH
    return pd.read_csv(path)


def to_dataset(df: pd.DataFrame) -> Dataset:
    numerical = [c for c in df.columns if c != TARGET]
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=[TARGET])
    return Dataset.from_pandas(df, data_definition=definition)


def main() -> None:
    df = load_data()
    reference, current = train_test_split(df, test_size=0.3, stratify=df[TARGET], random_state=42)

    report = Report([DataSummaryPreset(), DataDriftPreset()])
    snapshot = report.run(to_dataset(current), to_dataset(reference))

    EVIDENTLY_WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
    workspace = Workspace.create(str(EVIDENTLY_WORKSPACE_PATH))
    project = next(iter(workspace.search_project(PROJECT_NAME)), None)
    if project is None:
        project = workspace.create_project(PROJECT_NAME)
        project.description = "Data summary et data drift du dataset SIRTUIN6"
        project.save()
    workspace.add_run(project.id, snapshot)
    print(f"Rapport ajouté au workspace {EVIDENTLY_WORKSPACE_PATH}")


if __name__ == "__main__":
    main()
