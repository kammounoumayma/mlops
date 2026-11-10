
# ============================================================
# TP4 - MLflow Model Registry
# Chargement du modèle champion et prédiction d'un nouveau client
# ============================================================

# ===== 1. Importer les bibliothèques =====

import mlflow
import mlflow.sklearn
import pandas as pd


# ===== 2. Se connecter à la base MLflow =====

# Utiliser la même base de données que le TP3
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Charger le modèle champion =====

# Charger le modèle directement à partir de son alias
# Sans connaître sa version ni son run_id
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

print("Modèle chargé :", type(model).__name__)

print(
    "Variables attendues :",
    list(model.feature_names_in_)
)


# ===== 4. Créer un nouveau client =====

# Exemple : client avec un abonnement Standard
# Les variables d'abonnement sont déjà encodées en 0/1

client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])


# ===== 5. Préparer les données =====

# Conserver uniquement les variables attendues
# et respecter leur ordre dans le modèle

client = client[list(model.feature_names_in_)]


# ===== 6. Effectuer la prédiction =====

# 1 = Le client va partir
# 0 = Le client reste

prediction = model.predict(client)[0]

print(
    "Le client va-t-il partir ? ->",
    "OUI" if prediction == 1 else "NON"
)
