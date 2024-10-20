import pickle
from pathlib import Path
from typing import Dict

from fastapi import FastAPI

ROOT_DIR = Path(__file__).absolute().parent
MODEL_DIR = ROOT_DIR.joinpath("model")

MODEL_FILE = "model_C=1.0.bin"

MODEL_PATH = MODEL_DIR.joinpath(MODEL_FILE)

with MODEL_PATH.open("rb") as f_in:
    dv, model = pickle.load(f_in)


app = FastAPI()


@app.post("/predict")
def predict(customer: Dict):
    customer_dict = customer
    X = dv.transform([customer_dict])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn),
    }
    return result
