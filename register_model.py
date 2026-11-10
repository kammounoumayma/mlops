
# ============================================================
# TP4 - MLflow Model Registry
# Enregistrement du meilleur modèle et attribution d'un alias
# ============================================================

# ===== 1. Importer les bibliothèques =====

import mlflow
from mlflow import MlflowClient


# ===== 2. Se connecter à la base MLflow =====

# Utiliser la même base de données que le TP3
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Retrouver le meilleur run =====

# Trier les runs selon le score F1 décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Sélectionner le meilleur run
best = runs.iloc[0]

# Récupérer son identifiant
run_id = best["run_id"]

print(
    "Meilleur run :",
    best["tags.mlflow.runName"],
    "| id :",
    run_id[:8],
    "..."
)


# ===== 4. Enregistrer le modèle dans le Registry =====

# Chemin du modèle associé au meilleur run
model_uri = f"runs:/{run_id}/model"

# Nom du modèle dans le Registry
model_name = "churn-model"

# Enregistrer le modèle
result = mlflow.register_model(
    model_uri,
    model_name
)

print(
    "Modèle enregistré :",
    model_name,
    "| version :",
    result.version
)


# ===== 5. Attribuer l'alias "champion" =====

# Initialiser le client MLflow
client = MlflowClient()

# Associer l'alias champion à la version enregistrée
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=result.version
)

print(
    "Alias 'champion' -> version",
    result.version
)
