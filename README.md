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
| **Reproductibilité** | Seed unique (`RANDOM_STATE = 42`) partout (split, CV, bootstrap, bruit), version de Python figée (`.python-version`), dépendances figées à l'identique dans [requirements.txt](requirements.txt), l'image Airflow et l'image Evidently. Seules les données brutes sont versionnées : tout le reste se régénère, au bit près, depuis un clone neuf (voir [Reproduire le projet](#reproduire-le-projet)). |
| **Configuration centralisée** | Aucune valeur magique dans le code : chemins dans [config/settings.py](config/settings.py), paramètres ML et seuils qualité dans [config/ml_params.py](config/ml_params.py). |
| **Séparation logique métier / orchestration** | Toute la logique vit dans `src/` ; le DAG Airflow ne fait qu'enchaîner des appels. Chaque module est exécutable seul (bouton play ou `python -m ...`). |
| **Data validation** | `validate_data` bloque le pipeline si le jeu a trop peu de lignes, des NaN, ou des classes trop déséquilibrées. Les types sont forcés à l'extraction. |
| **Pas de fuite de données** | Le préprocesseur est ajusté uniquement sur le train (`learned_stats.json`) ; le modèle final est un `Pipeline` (préprocesseur + estimateur) unique et versionné. |
| **Évaluation rigoureuse** | Optimisation par CV répétée (5×3, ROC AUC), benchmark sur un jeu de test tenu à l'écart, puis tests de robustesse (voir ci-dessous). |
| **Quality gate** | Un modèle n'atteint le registre que si son AUC test **et** son AUC moyen en CV répétée dépassent `STAGING_MIN_ROC_AUC` (0.85). |
| **Registre et traçabilité** | Chaque modèle est logué dans MLflow (params, métriques, dataset d'entraînement, signature) et enregistré comme nouvelle version de `sirtuin6-<modèle>`. La version finale est reliée à sa version évaluée (`candidate_version`) et à son rapport Evidently (`evidently_snapshot`). |
| **Promotion progressive** | `candidate` (modèle évalué sur le test) → `staging` (modèle réentraîné sur tout le dataset). L'alias `production` n'est **jamais** posé automatiquement : c'est une décision humaine après revue dans MLflow. |
| **Smoke test consommateur** | Après mise en staging, le modèle est rechargé via `models:/sirtuin6-elastic@staging` comme le ferait un client, et doit prédire sur des données réelles. |
| **Monitoring des données** | Evidently produit des rapports de data summary et de data drift, consultables dans une UI dédiée. |
| **Infra as code** | Chaque service (MLflow, Airflow, Evidently) est décrit par un `docker-compose.yml` versionné. |
| **Secrets hors du dépôt** | Les secrets Airflow vivent dans `services/airflow/.env` (ignoré par git), créé depuis `.env.example`. |

## Vue d'ensemble

Le projet se déroule en deux temps.

**1. Lab (manuel, script par script)** : comparer trois modèles et choisir le champion.

```
 SIRTUIN6.csv ──► extract + validate ──► process (par modèle) ──► optimize (CV répétée)
                                                                        │
                 build ──► benchmark (jeu de test) ──► tests de robustesse ◄┘
                                     │
                          champion : elastic
```

**2. Production du champion (Airflow, DAG `sirtuin6_elastic_staging`)** : chaque lundi à 3h.

```
 extract + validate ──► process (split) ──► optimize ──► build ──► evaluate (test)
                                                                        │
                                                          quality gate (AUC ≥ 0.85)
                                                                        │
                                MLflow : nouvelle version, alias `candidate`
                                                                        │
                                 Evidently : rapport train vs test (tagué vN)
                                                                        │
                          réentraînement sur l'intégralité du dataset (train_final)
                                                                        │
                                MLflow : nouvelle version, alias `staging`
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

Résultats du benchmark sur le jeu de test (`data/artifacts/benchmark.csv`, généré par
`src.lab.benchmarking`) :

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
data/                       Seul raw/ est versionné, le reste est régénéré par les scripts
  raw/                      Données brutes (SIRTUIN6.csv)
  processed/                Données nettoyées et validées
  artifacts/<modèle>/       Splits, préprocesseur ajusté, meilleurs hyperparamètres
  artifacts/robustness/     Résultats des tests de robustesse
  artifacts/benchmark.csv   Comparaison des modèles sur le test
  models/                   Modèles sérialisés (joblib)
  figures/                  Graphiques de l'analyse descriptive
src/
  notebooks/                Analyse exploratoire (notebook)
  pipeline/                 extract_data, process_data, build_models, train_final
  lab/                      hyperparameters_optimizations, benchmarking
  robustness/               Cross-validation répétée, bootstrap, bruit gaussien,
                            sensibilité aux hyperparamètres
  monitoring/               mlflow_tracking, mlflow_staging, evidently_reports
services/
  mlflow/                   Serveur MLflow (tracking + registre), port 5001
  airflow/                  Airflow + DAG `sirtuin6_elastic_staging`, port 8080
  evidently/                UI Evidently, port 8000
.env                        PYTHONPATH=. pour le bouton play (VS Code)
.python-version             Version de Python du projet
requirements.txt            Dépendances Python figées
```

## Robustesse

Au-delà d'un simple score de test, quatre analyses ([src/robustness/](src/robustness/)) mesurent
la fiabilité de chaque modèle (ROC AUC, accuracy, balanced accuracy, F1) :

- **Validation croisée répétée** (5 plis × 10 répétitions) : moyenne et dispersion.
- **Bootstrap** (200 rééchantillonnages) : intervalles de confiance à 95 %.
- **Bruit gaussien** : dégradation des performances quand on bruite les variables
  (0 → 100 % de leur écart-type).
- **Sensibilité aux hyperparamètres** : la performance dépend-elle fortement du réglage retenu ?

## Reproduire le projet

Prérequis : **Python 3.13**, **Docker**, VS Code (recommandé, pour le bouton play).

### 1. Environnement Python

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dans VS Code, sélectionner l'interpréteur `.venv`. Le fichier `.env` racine (`PYTHONPATH=.`) et
`.vscode/settings.json` rendent `config` et `src` importables : chaque script se lance
directement avec le **bouton play**. En ligne de commande, lancer depuis la racine avec
`python -m <module>`.

### 2. Services

```bash
# MLflow  -> http://localhost:5001
docker compose -f services/mlflow/docker-compose.yml up -d

# Evidently -> http://localhost:8000
docker compose -f services/evidently/docker-compose.yml up -d --build
```

### 3. Lab : scripts à lancer un par un, dans cet ordre

| # | Script | Produit |
|---|---|---|
| 0 | `src/notebooks/descriptive_analyses.ipynb` (optionnel) | `data/figures/*.png` |
| 1 | `src.pipeline.extract_data` | `data/processed/sirtuin6_clean.csv` |
| 2 | `src.pipeline.process_data` | `data/artifacts/<modèle>/` (splits, préprocesseurs) |
| 3 | `src.lab.hyperparameters_optimizations` | `data/artifacts/<modèle>/best_params.json` |
| 4 | `src.pipeline.build_models` | `data/models/<modèle>.joblib` |
| 5 | `src.lab.benchmarking` | `data/artifacts/benchmark.csv` → choix du champion |
| 6 | `src.robustness.cross_validations`, `bootstraps`, `gaussian_noises`, `hyperparameters_sensitivities` | `data/artifacts/robustness/*.csv` |
| 7 | `src.pipeline.train_final` | `data/models/final_<champion>.joblib` |
| 8 | `src.monitoring.mlflow_tracking` (MLflow démarré) | une version par modèle dans le registre |
| 9 | `src.monitoring.mlflow_staging` | alias `staging` sur le champion |
| 10 | `src.monitoring.evidently_reports` | rapport dans le workspace Evidently |

Les résultats sont déterministes : un clone neuf retrouve exactement les mêmes splits,
hyperparamètres, métriques et prédictions.

### 4. Airflow : DAG du champion elastic

```bash
cd services/airflow
cp .env.example .env
# remplir FERNET_KEY (commande de génération dans le fichier) et _AIRFLOW_WWW_USER_PASSWORD
docker compose up -d --build
```

Interface : http://localhost:8080, identifiants `_AIRFLOW_WWW_USER_USERNAME` /
`_AIRFLOW_WWW_USER_PASSWORD` du `.env`. Activer puis lancer `sirtuin6_elastic_staging`.

- Le dépôt est monté dans les conteneurs (`PYTHONPATH=/opt/airflow/project`) : le DAG importe
  directement `src/` et `config/`, et écrit dans les mêmes `data/`, workspace Evidently et
  registre MLflow (via `host.docker.internal:5001`) que les scripts locaux.
- Le compte admin n'est créé qu'au **premier** passage d'`airflow-init`. Pour changer ensuite
  le mot de passe : `docker exec airflow-airflow-apiserver-1 airflow users reset-password -u admin -p '<mdp>'`.

**Deux fichiers `.env` :**

- `.env` (racine) : `PYTHONPATH=.`, lu par VS Code (`python.envFile`). Aucun secret, versionné.
- `services/airflow/.env` : lu par `docker compose` d'Airflow (`AIRFLOW_UID`, `FERNET_KEY`,
  identifiant / mot de passe admin). Contient des secrets : **non versionné**.

Cette configuration est prévue pour le **développement local**, pas pour la production.

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
