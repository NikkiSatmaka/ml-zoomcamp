import numpy as np
import polars as pl
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder


def prepare_X(
    df: pl.DataFrame,
    encoder: OneHotEncoder,
    cat_cols: list,
    num_cols: list,
) -> np.ndarray:
    X_cat: np.ndarray = encoder.transform(df[cat_cols])
    X_num: np.ndarray = df[num_cols].to_numpy()
    X: np.ndarray = np.hstack((X_cat, X_num))
    return X


def get_churn_score_pipeline(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cat_cols: list,
    num_cols: list,
    C: float = 1.0,
) -> np.float64:
    ohe: OneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(df_train[cat_cols])

    X_train: np.ndarray = prepare_X(df_train, ohe, cat_cols, num_cols)
    X_val: np.ndarray = prepare_X(df_val, ohe, cat_cols, num_cols)

    model = LogisticRegression(
        solver="liblinear",
        C=C,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    churn_decision = y_pred >= 0.5
    accuracy = (y_val == churn_decision).mean()
    return accuracy


def get_churn_score_dv_pipeline(
    df: pl.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cat_cols: list,
    num_cols: list,
    C: float = 1.0,
    seed: int = 42,
) -> np.float64:
    df_train = df[train_idx]
    df_val = df[val_idx]

    y_train = df_train["y"].to_numpy()
    y_val = df_val["y"].to_numpy()

    assert len(df_train) == len(y_train)
    assert len(df_val) == len(y_val)

    dicts_train = df_train.select(pl.col(cat_cols + num_cols)).to_dicts()
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts_train, y_train)

    dicts_val = df_val.select(pl.col(cat_cols + num_cols)).to_dicts()
    X_val = dv.transform(dicts_val)

    model = LogisticRegression(
        solver="liblinear",
        C=C,
        max_iter=1000,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred)
    return auc_score
