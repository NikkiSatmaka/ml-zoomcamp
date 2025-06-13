import re
from pathlib import Path

import polars as pl


def load_data(source, data_dir: Path, **kwargs) -> pl.DataFrame:
    stem = Path(source).stem
    suffix = Path(source).suffix
    name_parquet = stem + ".parquet"
    path_parquet = data_dir.joinpath(name_parquet)

    def remove_csv_kwargs(kwargs):
        kwargs.pop("separator", None)
        kwargs.pop("has_header", None)
        return kwargs

    if path_parquet.exists():
        kwargs = remove_csv_kwargs(kwargs)
        df = pl.read_parquet(path_parquet, **kwargs)
    elif suffix == ".parquet":
        kwargs = remove_csv_kwargs(kwargs)
        df = pl.read_parquet(source, **kwargs)
        df.write_parquet(path_parquet)
    elif suffix == ".csv":
        df = pl.read_csv(source, **kwargs)
        df.write_parquet(path_parquet)
    elif suffix == ".json":
        kwargs = remove_csv_kwargs(kwargs)
        df = pl.read_json(source, **kwargs)
        df.write_parquet(path_parquet)
    elif suffix == ".jsonl":
        kwargs = remove_csv_kwargs(kwargs)
        df = pl.read_json(source, **kwargs)
        df.write_parquet(path_parquet)
    elif suffix == ".tsv":
        df = pl.read_csv(source, separator="\t", **kwargs)
        df.write_parquet(path_parquet)
    else:
        df = pl.read_csv(source, **kwargs)
        df.write_parquet(path_parquet)

    return df


def clean_alphanumeric(s: str) -> str:
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


def normalize_name(s: str) -> str:
    return clean_alphanumeric(s).lower()


def clean_column_names(df: pl.DataFrame) -> pl.DataFrame:
    df.columns = [*map(normalize_name, df.columns)]
    return df
