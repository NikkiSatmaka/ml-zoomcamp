import pickle
from pathlib import Path
from typing import Dict

from fastapi import FastAPI

ROOT_DIR = Path(__file__).absolute().parent
MODEL_DIR = ROOT_DIR.joinpath("model")

DV_FILE = "dv.bin"
MODEL_FILE = "model1.bin"

DV_PATH = MODEL_DIR.joinpath(DV_FILE)
MODEL_PATH = MODEL_DIR.joinpath(MODEL_FILE)


def load(filepath: Path):
    with filepath.open("rb") as f_in:
        return pickle.load(f_in)


DV = load(DV_PATH)
MODEL = load(MODEL_PATH)

app = FastAPI()


@app.post("/predict")
def predict(client: Dict):
    client_dict = client

    X = DV.transform([client_dict])
    y_pred = MODEL.predict_proba(X)[0, 1]
    subscribe = y_pred >= 0.5

    result = {
        "subscription_probability": float(y_pred),
        "subscribe": bool(subscribe),
    }
    return result
