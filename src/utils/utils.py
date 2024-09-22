from pathlib import Path

import polars as pl


def load_data(csv_uri, data_dir):
    stem = Path(csv_uri).stem
    name_parquet = stem + ".parquet"
    path_parquet = data_dir.joinpath(name_parquet)

    if path_parquet.exists():
        df = pl.read_parquet(path_parquet)
    else:
        df = pl.read_csv(csv_uri)
        df.write_parquet(path_parquet)

    return df
