#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path

from loguru import logger

ROOT_DIR = Path(__file__).absolute().parent.parent.parent
DATA_DIR = ROOT_DIR.joinpath("data")
MODEL_DIR = ROOT_DIR.joinpath("model")

MODEL_FILE = "model_C=1.0.bin"

MODEL_PATH = MODEL_DIR.joinpath(MODEL_FILE)

CUSTOMER = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "tenure": 1,
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month_to_month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}


def main():
    with MODEL_PATH.open("rb") as f_in:
        dv, model = pickle.load(f_in)

    X = dv.transform([CUSTOMER])
    y_pred = model.predict_proba(X)[0, 1]

    logger.info(f"Input: {CUSTOMER}")
    logger.info(f"Churn probability: {y_pred}")


if __name__ == "__main__":
    main()
