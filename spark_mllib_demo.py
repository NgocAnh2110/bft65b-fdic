from pathlib import Path

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.sql import SparkSession, functions as F

project = Path(__file__).resolve().parent
source = project / "data_processed" / "feature_table.parquet"
out = project / "outputs"
out.mkdir(exist_ok=True)

spark = (
    SparkSession.builder.master("local[2]")
    .appName("BFT65B-MLlib-development")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

try:
    fields = [
        "log_assets_lag1", "equity_assets_lag1", "deposits_assets_lag1",
        "reserves_loans_lag1", "roa_lag1",
    ]
    filled = [f"{name}_filled" for name in fields]

    dev = (
        spark.read.parquet(source.as_uri())
        .where(F.col("split") == "development")
        .select(
            "REPDTE", *fields,
            F.col("source_report").isNull().cast("double").alias("no_prior_report"),
            F.col("y").cast("double").alias("label"),
        )
    )
    train = dev.where(F.col("REPDTE") <= F.lit("2001-12-31").cast("date"))
    valid = dev.where(
        F.col("REPDTE").between(
            F.lit("2003-03-31").cast("date"),
            F.lit("2007-12-31").cast("date"),
        )
    )

    # The pipeline learns imputation values and model parameters on train only.
    pipeline = Pipeline(stages=[
        Imputer(strategy="median", inputCols=fields, outputCols=filled),
        VectorAssembler(inputCols=filled + ["no_prior_report"], outputCol="features"),
        LogisticRegression(
            featuresCol="features", labelCol="label",
            regParam=0.01, elasticNetParam=0.0,
            standardization=True, maxIter=100,
        ),
    ])
    fitted = pipeline.fit(train)
    scored = fitted.transform(valid).select("label", "rawPrediction").cache()

    counts = scored.agg(
        F.count("*").alias("validation_rows"),
        F.sum("label").alias("positive_rows"),
    ).first()
    if (counts["validation_rows"], int(counts["positive_rows"])) != (180160, 74):
        raise ValueError("MLlib validation rows differ from the audited third fold.")

    pr_auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ).evaluate(scored)

    result = pd.DataFrame([{
        "spark_version": spark.version,
        "train_end": "2001-12-31",
        "validation_rows": counts["validation_rows"],
        "positive_rows": int(counts["positive_rows"]),
        "spark_area_under_PR": pr_auc,
    }])
    result.to_csv(out / "spark_mllib_demo.csv", index=False)
    print("\nMLlib development demo:")
    print(result.to_string(index=False))
    scored.unpersist()
finally:
    spark.stop()
