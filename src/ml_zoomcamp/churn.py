import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
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
    ohe: OneHotEncoder = OneHotEncoder(sparse_output=False)
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
