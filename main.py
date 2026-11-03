from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import numpy as np
import os

# MLFlow Configuration
MLFLOW_TRACKING_URI=os.getenv(
    "MLFLOW_TRACKING_URI","http://127.0.0.1:5000"
)

# models/iris-classifier@champion
MODEL_URI="models:/iris-classifier@champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#MODEL_PATH = Path("models/iris_random_forest_model.joblib")
#model = joblib.load(MODEL_PATH)

try:
    # Load the champion model at application startup
    model = mlflow.sklearn.load_model(MODEL_URI)
    client=mlflow.MlflowClient()
    champion_version=client.get_model_version_by_alias(
        name="iris-classifier",
        alias="champion"
    )
    model_version=champion_version.version
    print(f"Loaded Models: Iris Classifier version: {model_version}")

except Exception as e:
    raise RuntimeError(f"Couldn't load champion model from mlflow: {e}")

# Create FastAPI Application
app = FastAPI(title="Iris Random Forest API")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def ready_to_serve():
    return {""
        "status": "Ready to serve",
        "model_name":"iris-classifier",
        "model_version":model_version
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Endpoint for making predictions with the Iris Random Forest model
@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = int(model.predict(features)[0])
    print("Prediction is",prediction)
    return {
        "prediction": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction]
    }