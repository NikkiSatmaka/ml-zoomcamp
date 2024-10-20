#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path

from flask import Flask, jsonify, request

ROOT_DIR = Path(__file__).absolute().parent.parent.parent
MODEL_DIR = ROOT_DIR.joinpath("model")

MODEL_FILE = "model_C=1.0.bin"

MODEL_PATH = MODEL_DIR.joinpath(MODEL_FILE)

with MODEL_PATH.open("rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("churn")


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn),
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
