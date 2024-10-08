import numpy as np
import polars as pl


def train_val_test_split(
    df: pl.DataFrame,
    y_col: str,
    seed: int = 42,
    train_val_test_ratio: tuple = (0.6, 0.2, 0.2),
) -> tuple:
    assert sum(train_val_test_ratio) == 1
    r_train, r_val, r_test = train_val_test_ratio
    n = df.shape[0]

    n_val = int(n * r_val)
    n_test = int(n * r_test)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df[idx[:n_train]]
    df_val = df[idx[n_train : n_train + n_val]]
    df_test = df[idx[n_train + n_val :]]

    assert df[idx].equals(df_train.vstack(df_val).vstack(df_test))

    y_train = df_train.select(pl.col(y_col)).to_numpy().flatten()
    y_val = df_val.select(pl.col(y_col)).to_numpy().flatten()
    y_test = df_test.select(pl.col(y_col)).to_numpy().flatten()

    df_train = df_train.drop(y_col)
    df_val = df_val.drop(y_col)
    df_test = df_test.drop(y_col)

    return df_train, df_val, df_test, y_train, y_val, y_test


def prepare_X(df: pl.DataFrame, null_replacement: float) -> np.ndarray:
    df = df.with_columns(pl.col("screen").fill_null(null_replacement))
    X = df.to_numpy()
    return X


def train_linear_regression(X: np.ndarray, y: np.ndarray, r: float = 0) -> tuple:
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T @ X
    XTX += r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv @ X.T @ y

    return w_full[0], w_full[1:]


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)
