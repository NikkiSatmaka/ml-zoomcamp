import polars as pl


def parse_xgb_output(output, metric):
    results = []

    for line in output.stdout.strip().split("\n"):
        it_line, train_line, val_line = line.split("\t")

        it = int(it_line.strip("[]"))
        train = float(train_line.split(":")[1])
        val = float(val_line.split(":")[1])

        results.append((it, train, val))

    columns = ["num_iter", f"train_{metric}", f"val_{metric}"]
    df_results = pl.DataFrame(results, schema=columns, orient="row")
    return df_results
