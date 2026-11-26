from datetime import date

import pandas as pd
import polars as pl

PANEL = "data_processed/financial_panel.parquet"
LABELS = "data_processed/modelling_labels.parquet"
START = date(2002, 3, 31)
END = date(2007, 12, 31)
SOURCE_START = date(2001, 12, 31)
SOURCE_END = date(2007, 9, 30)
KEY = ["CERT", "q"]


def pandas_join():
    targets = pd.read_parquet(LABELS, columns=["CERT", "REPDTE"])
    reports = pd.read_parquet(PANEL, columns=["CERT", "REPDTE", "EQ", "ASSET"])

    for frame in (targets, reports):
        frame["REPDTE"] = pd.to_datetime(frame["REPDTE"])
        frame["q"] = frame["REPDTE"].dt.year * 4 + frame["REPDTE"].dt.quarter

    targets = targets.loc[
        targets["REPDTE"].between(pd.Timestamp(START), pd.Timestamp(END)), KEY
    ]
    reports = reports.loc[
        reports["REPDTE"].between(pd.Timestamp(SOURCE_START), pd.Timestamp(SOURCE_END))
    ].copy()

    reports["ratio"] = reports["EQ"] / reports["ASSET"].where(reports["ASSET"] > 0)
    reports["q"] += 1
    source = reports.rename(columns={"REPDTE": "source_date"})
    return targets.merge(
        source[KEY + ["source_date", "ratio"]],
        on=KEY, how="left", validate="one_to_one"
    )


def polars_join():
    day = pl.col("REPDTE")
    quarter = day.dt.year() * 4 + day.dt.quarter().cast(pl.Int32)

    targets = (
        pl.scan_parquet(LABELS)
        .filter(day.is_between(START, END))
        .select("CERT", quarter.alias("q"))
    )
    source = (
        pl.scan_parquet(PANEL)
        .filter(day.is_between(SOURCE_START, SOURCE_END))
        .select(
            "CERT",
            (quarter + 1).alias("q"),
            day.alias("source_date"),
            pl.when(pl.col("ASSET") > 0)
              .then(pl.col("EQ") / pl.col("ASSET"))
              .alias("ratio"),
        )
    )
    return targets.join(source, on=KEY, how="left", validate="1:1").collect()


if __name__ == "__main__":
    left = pandas_join().sort_values(KEY).reset_index(drop=True)
    right = polars_join().to_pandas().sort_values(KEY).reset_index(drop=True)

    for frame in (left, right):
        frame["source_date"] = pd.to_datetime(frame["source_date"]).astype("datetime64[ns]")

    pd.testing.assert_frame_equal(
        left, right, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12
    )

    print(f"Prediction quarters: {left['q'].nunique():,}")
    print(f"Bank-quarters compared: {len(left):,}")
    print(f"Missing prior-quarter reports: {left['source_date'].isna().sum():,}")
    print(f"Missing equity/assets ratios: {left['ratio'].isna().sum():,}")
    print("Engine comparison: PASS")

