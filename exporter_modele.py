
import mlflow
import mlflow.sklearn

# Connexion au serveur MLflow Docker
mlflow.set_tracking_uri("http://localhost:5000")

# Chargement du modele champion
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

# Exportation du modele
mlflow.sklearn.save_model(
    model,
    "model",
    serialization_format="cloudpickle"
)

print("Modele exporte avec succes !")
