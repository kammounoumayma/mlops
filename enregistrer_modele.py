
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from mlflow.tracking import MlflowClient

# Connexion au serveur MLflow Docker
mlflow.set_tracking_uri("http://localhost:5000")

# Creation de l'experience
mlflow.set_experiment("churn-docker")

# Chargement des donnees
df = pd.read_csv("churn_data.csv")

# Preparation des variables
X = pd.get_dummies(
    df.drop(columns=["churn"]),
    columns=["abonnement"]
)

y = df["churn"]

# Entrainement du modele
modele = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modele.fit(X, y)

# Enregistrement dans MLflow
with mlflow.start_run():
    model_info = mlflow.sklearn.log_model(
        modele,
        name="model",
        registered_model_name="churn-model",
        serialization_format="cloudpickle"
    )

# Attribution de l'alias champion
client = MlflowClient()

client.set_registered_model_alias(
    "churn-model",
    "champion",
    model_info.registered_model_version
)

print("Modele enregistre avec succes !")
print("Alias champion attribue.")
