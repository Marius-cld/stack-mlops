# stack-ml

Projet de démonstration d'une **stack MLOps locale** de bout en bout, appliquée à un problème de
classification binaire en chimie : prédire si une petite molécule a une énergie de liaison
élevée (`High_BFE`) ou faible (`Low_BFE`) sur la protéine **Sirtuin 6**.

Le sujet ML est volontairement modeste (petit jeu tabulaire) : l'intérêt du dépôt est la
**chaîne industrielle** autour du modèle — données validées, entraînement reproductible,
tests de robustesse, registre de modèles, orchestration, monitoring et promotion contrôlée.

## Culture MLOps : principes appliqués

| Principe | Comment il est appliqué ici |
|---|---|
| **Reproductibilité** | Seed unique (`RANDOM_STATE = 42`), découpage stratifié, hyperparamètres et grilles centralisés dans [config/ml_params.py](config/ml_params.py), versions de scikit-learn/pandas/mlflow figées dans l'image Airflow. |
| **Configuration centralisée** | Aucune valeur magique dans le code : chemins dans [config/setting.py](config/setting.py), paramètres ML et seuils qualité dans [config/ml_params.py](config/ml_params.py). |
| **Séparation logique métier / orchestration** | Toute la logique vit dans `src/` ; le DAG Airflow ne fait qu'enchaîner des appels. Chaque module est exécutable seul (`python -m ...`). |
| **Data validation** | `validate_data` bloque le pipeline si le jeu a trop peu de lignes, des NaN, ou des classes trop déséquilibrées. Les types sont forcés à l'extraction. |
| **Pas de fuite de données** | Le préprocesseur est ajusté uniquement sur le train (`learned_stats.json`) ; le modèle final est un `Pipeline` (préprocesseur + estimateur) unique et versionné. |
| **Évaluation rigoureuse** | Optimisation par CV répétée (5×3, ROC AUC), benchmark sur un jeu de test tenu à l'écart, puis tests de robustesse (voir ci-dessous). |
| **Quality gate** | Un modèle n'atteint le registre que si son AUC test **et** son AUC moyen en CV répétée dépassent `STAGING_MIN_ROC_AUC` (0.85). |
| **Registre et traçabilité** | Chaque entraînement final est logué dans MLflow (params, métriques, dataset d'entraînement, signature) et enregistré comme nouvelle version de `sirtuin6-<modèle>`. |
| **Promotion progressive** | Le pipeline automatisé pose seulement l'alias `staging`. L'alias `production` n'est **jamais** posé automatiquement : c'est une décision humaine après revue dans MLflow. |
| **Smoke test consommateur** | Après mise en staging, le modèle est rechargé via `models:/sirtuin6-<modèle>@staging` comme le ferait un client, et doit prédire sur des données réelles. |
| **Monitoring des données** | Evidently produit des rapports de data summary et de data drift, consultables dans une UI dédiée. |
| **Infra as code** | Chaque service (MLflow, Airflow, Evidently) est décrit par un `docker-compose.yaml` versionné. |

## Vue d'ensemble

```
 SIRTUIN6.csv ──► extract + validate ──► process (par modèle) ──► optimize (CV répétée)
                                                                        │
                       ┌────────────────────────────────────────────────┘
                       ▼
                 build ──► benchmark (jeu de test) ──► meilleur modèle
                                                             │
                                                    quality gate (AUC ≥ 0.85)
                                                             │
                                          entraînement final sur toutes les données
                                                             │
                                    MLflow : log + registre ──► alias `staging`
                                                             │
                                                        smoke test
                                                             │
                                     revue humaine ──► alias `production` (manuel)
```

Les trois modèles comparés :

| Nom | Estimateur | Préprocessing |
|---|---|---|
| `elastic` | Régression logistique elastic-net (saga) | Yeo-Johnson sur les variables asymétriques + standardisation |
| `svm` | SVM à noyau RBF | Standardisation |
| `trees` | Random Forest (300 arbres) | Aucun |

Résultats du benchmark sur le jeu de test ([data/artifacts/benchmark.csv](data/artifacts/benchmark.csv)) :

| Modèle | ROC AUC | Accuracy | F1 |
|---|---|---|---|
| elastic | 0.94 | 0.90 | 0.89 |
| svm | 0.93 | 0.90 | 0.89 |
| trees | 0.91 | 0.80 | 0.78 |

> Le jeu de test est petit (20 lignes) : ces chiffres sont indicatifs, d'où les tests de
> robustesse ci-dessous.

## Structure du dépôt

```
config/                     Chemins et paramètres ML (source unique de vérité)
data/
  raw/                      Données brutes (SIRTUIN6.csv)
  processed/                Données nettoyées et validées
  artifacts/<modèle>/       Splits, préprocesseur ajusté, meilleurs hyperparamètres
  artifacts/robustness/     Résultats des tests de robustesse
  artifacts/benchmark.csv   Comparaison des modèles sur le test
  models/                   Modèles sérialisés (joblib)
  figures/                  Graphiques de l'analyse descriptive
src/
  descriptive_statistics/   Analyse exploratoire (script + notebook)
  pipeline/                 extract_data, process_data, build_models, train_final
  lab/                      hyperparameters_optimizations, benchmarking
  robustness/               Cross-validation répétée, bootstrap, bruit gaussien,
                            sensibilité aux hyperparamètres
  monitoring/               mlflow_tracking, mlflow_staging, evidently_reports
services/
  mlflow/                   Serveur MLflow (tracking + registre), port 5001
  airflow/                  Airflow + DAG `sirtuin6_staging_model`
  evidentlyia/              UI Evidently, port 8000
```

## Robustesse

Au-delà d'un simple score de test, quatre analyses ([src/robustness/](src/robustness/)) mesurent
la fiabilité de chaque modèle (ROC AUC, accuracy, balanced accuracy, F1) :

- **Validation croisée répétée** (5 plis × 10 répétitions) : moyenne et dispersion.
- **Bootstrap** (200 rééchantillonnages) : intervalles de confiance à 95 %.
- **Bruit gaussien** : dégradation des performances quand on bruite les variables
  (0 → 100 % de leur écart-type).
- **Sensibilité aux hyperparamètres** : la performance dépend-elle fortement du réglage retenu ?

## Démarrage

Prérequis : Python 3, Docker.

```bash
pip install -r requierments.txt        # + pandas, scikit-learn, evidently, etc.

# 1. Serveur MLflow  -> http://localhost:5001
docker compose -f services/mlflow/docker-compose.yaml up -d

# 2. Pipeline complet en local (depuis la racine du dépôt)
python -m src.pipeline.extract_data
python -m src.lab.hyperparameters_optimizations
python -m src.pipeline.build_models
python -m src.lab.benchmarking
python -m src.monitoring.mlflow_tracking     # log + enregistre dans le registre
python -m src.monitoring.mlflow_staging      # alias staging

# 3. Tests de robustesse (optionnel)
python -m src.robustness.cross_validations
python -m src.robustness.bootstraps
python -m src.robustness.gaussian_noises
python -m src.robustness.hyperparameters_sensitivities

# 4. Monitoring des données -> UI http://localhost:8000
python -m src.monitoring.evidently_reports
docker compose -f services/evidentlyia/docker-compose.yaml up -d --build
```

### Orchestration avec Airflow

Le DAG [staging_model.py](services/airflow/dag/staging_model.py) (`sirtuin6_staging_model`)
exécute tout le pipeline chaque lundi à 3h (Europe/Paris) : validation des données →
process / optimisation / build en parallèle par modèle → benchmark → quality gate →
enregistrement → alias `staging` → smoke test.

```bash
cd services/airflow
docker compose up -d --build
```

Le dépôt est monté dans les conteneurs (`PYTHONPATH=/opt/airflow/project`), donc le DAG importe
directement `src/` et `config/`. Le fichier `services/airflow/.env` contient `FERNET_KEY` : à
régénérer et à ne pas publier pour un usage réel. Cette configuration est prévue pour le
**développement local**, pas pour la production.

### Promouvoir en production

Après revue de la version `staging` dans l'UI MLflow, poser manuellement l'alias `production`
sur la version retenue. Les consommateurs chargent alors :

```python
from src.monitoring.mlflow_tracking import load_production
model = load_production("elastic")   # models:/sirtuin6-elastic@production
```

## Pistes d'amélioration

- Déclencher le ré-entraînement sur alerte de drift Evidently plutôt que sur un calendrier fixe.
- Ajouter des tests unitaires et une CI (lint, tests, validation des données).
- Versionner les données (DVC) et externaliser le stockage d'artefacts MLflow (S3/MinIO).
- Exposer le modèle `production` via une API de serving.

## Dataset

Les données utilisées dans ce projet (`data/raw/SIRTUIN6.csv`) proviennent de :

Tardu, M. & RAHIM, F. (2016). Sirtuin6 Small Molecules [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C9Z.

Le jeu de données est distribué par l'UCI Machine Learning Repository sous licence
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

## Licence

Le code source est publié sous licence MIT (voir [LICENSE](LICENSE)).
Le jeu de données reste sous sa licence d'origine (CC BY 4.0), voir ci-dessus.
