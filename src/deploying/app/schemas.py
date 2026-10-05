"""Contrat de l'API : schémas d'entrée et de sortie, publiés dans `/docs` et `/openapi.json`."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from config.ml_params import MAX_BATCH_SIZE

# Classes de la cible (`config.ml_params.CLASSES`), écrites en clair pour typer le contrat
Label = Literal["Low_BFE", "High_BFE"]


class Molecule(BaseModel):
    """Descripteurs moléculaires PaDEL d'une molécule, sous les noms des colonnes du dataset.

    Valeurs finies uniquement : NaN et infini sont refusés (422) au lieu de faire échouer le modèle.
    """

    # Un descripteur mal orthographié est refusé plutôt qu'ignoré silencieusement
    model_config = ConfigDict(extra="forbid")

    sc_5: FiniteFloat = Field(
        alias="SC-5",
        description="Indice de connectivité de Kier-Hall : cluster simple d'ordre 5",
        examples=[0.540936],
    )
    sp_6: FiniteFloat = Field(
        alias="SP-6",
        description="Indice de connectivité de Kier-Hall : chemin simple d'ordre 6",
        examples=[7.64192],
    )
    shbd: FiniteFloat = Field(
        alias="SHBd",
        description="Somme des E-States des donneurs forts de liaison hydrogène",
        examples=[0.162171],
    )
    min_haa_ch: FiniteFloat = Field(
        alias="minHaaCH",
        description="E-State minimal des hydrogènes portés par un CH aromatique",
        examples=[0.44527],
    )
    max_whba: FiniteFloat = Field(
        alias="maxwHBa",
        description="E-State maximal des accepteurs faibles de liaison hydrogène",
        examples=[2.20557],
    )
    fmf: FiniteFloat = Field(
        alias="FMF",
        description="Fraction des atomes lourds appartenant au squelette moléculaire (complexité)",
        examples=[0.467742],
    )

    @classmethod
    def features(cls) -> list[str]:
        """Noms des descripteurs attendus en entrée (alias des champs)."""
        return [field.alias or name for name, field in cls.model_fields.items()]


class BatchRequest(BaseModel):
    """Lot de molécules à prédire en un seul appel."""

    model_config = ConfigDict(extra="forbid")

    molecules: list[Molecule] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    """Prédiction pour une molécule."""

    label: Label = Field(description="Classe prédite : énergie de liaison élevée ou faible")
    probabilities: dict[Label, float] = Field(description="Probabilité par classe (somme = 1)")


class ModelRef(BaseModel):
    """Version du registre MLflow qui a produit la réponse (traçabilité)."""

    name: str = Field(examples=["sirtuin6-elastic"])
    version: str = Field(examples=["10"])
    alias: str = Field(examples=["production"])


class PredictionResponse(Prediction):
    """Prédiction d'une molécule et version du modèle qui l'a produite."""

    model: ModelRef


class BatchResponse(BaseModel):
    """Prédictions d'un lot et version du modèle qui les a produites."""

    predictions: list[Prediction] = Field(
        description="Une prédiction par molécule, dans l'ordre de la requête"
    )
    model: ModelRef


class ModelInfo(ModelRef):
    """Fiche du modèle servi."""

    run_id: str = Field(description="Run MLflow qui a produit cette version")
    features: list[str] = Field(description="Descripteurs attendus en entrée")
    classes: list[Label]
    positive_class: Label
    params: dict[str, str] = Field(description="Hyperparamètres retenus par l'optimisation")
    metrics: dict[str, float] = Field(description="Métriques loguées dans MLflow (test, CV)")


class Health(BaseModel):
    """État de l'API et modèle servi."""

    status: Literal["ok"]
    model: ModelRef
