#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from loguru import logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

from ml_zoomcamp.utils import clean_column_names, load_data

ROOT_DIR = Path(__file__).absolute().parent.parent.parent
DATA_DIR = ROOT_DIR.joinpath("data")
MODEL_DIR = ROOT_DIR.joinpath("model")

CSV_URI = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"

C = 1.0
N_SPLITS = 5

NUMERICAL = ["tenure", "monthlycharges", "totalcharges"]

CATEGORICAL = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]


def train(df, y_train, C=1.0):
    dicts = df.select(pl.col(CATEGORICAL + NUMERICAL)).to_dicts()

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df.select(pl.col(CATEGORICAL + NUMERICAL)).to_dicts()

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    ## Data Preparation

    df = load_data(CSV_URI, DATA_DIR)
    df = clean_column_names(df)

    df = df.with_columns(
        cs.string()
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s-]", "")
        .str.replace_all(r"\s+|-+", "_")
    )

    df = df.with_columns(
        pl.col("totalcharges").fill_null(0),
        (pl.col("churn") == "yes").cast(pl.Int8),
    )

    ## Setting Up Validation Framework

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    y_train = df_train["churn"].to_numpy()
    y_val = df_val["churn"].to_numpy()
    y_test = df_test["churn"].to_numpy()

    df_train = df_train.drop("churn")
    df_val = df_val.drop("churn")
    df_test = df_test.drop("churn")

    ## Cross-Validation
    logger.info(f"Doing validation with C={C}")

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1)

    scores = []

    fold = 0

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train[train_idx]
        df_val = df_full_train[val_idx]

        y_train = df_train["churn"].to_numpy()
        y_val = df_val["churn"].to_numpy()

        dv, model = train(df_train, y_train, C)
        y_pred = predict(df_val, dv, model)

        auc_score = roc_auc_score(y_val, y_pred)
        scores.append(auc_score)

        logger.debug(f"AUC on fold {fold} is {auc_score}")
        fold += 1

    logger.debug("Validation results:")
    logger.debug("C=%s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))

    dv, model = train(df_full_train, df_full_train["churn"].to_numpy(), C=1.0)
    y_pred = predict(df_test, dv, model)

    auc_score = roc_auc_score(y_test, y_pred)

    logger.debug(f"AUC={auc_score}")

    ## Save the model

    logger.info("Training the final model")

    output_path = MODEL_DIR.joinpath(f"model_C={C}.bin")

    with output_path.open("wb") as f_out:
        pickle.dump((dv, model), f_out)

    logger.info(f"The model is saved to {output_path}")


if __name__ == "__main__":
    main()
