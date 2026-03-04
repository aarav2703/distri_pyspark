import os
import json
import argparse
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# Keep Spark config minimal; do NOT copy the special Windows configs from 01 unless needed.
def build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[4]")  # limit concurrency to reduce heap pressure
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        # give JVM more heap (adjust if your RAM is lower)
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        # reduce Parquet batch memory spikes on Windows
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        # reduce shuffle fanout (dropDuplicates will shuffle)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.warehouse.dir", os.path.abspath("./spark-warehouse"))
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )


CANON = {"view": "view", "cart": "cart", "purchase": "purchase"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_bronze", required=True, help="Path to bronze parquet root")
    ap.add_argument("--out_silver", required=True, help="Path to silver parquet root")
    ap.add_argument(
        "--repartition", type=int, default=0, help="Optional repartition before write"
    )
    args = ap.parse_args()

    spark = build_spark("02_clean_silver_events")

    df = spark.read.parquet(args.in_bronze)

    # Normalize event_type to lower-case canonical set
    df = df.withColumn("event_type_norm", F.lower(F.col("event_type")))
    df = df.withColumn(
        "event_type",
        F.when(
            F.col("event_type_norm").isin(list(CANON.keys())), F.col("event_type_norm")
        ).otherwise(F.lit(None)),
    ).drop("event_type_norm")

    # Required fields for analytics
    df = df.filter(
        F.col("event_ts").isNotNull()
        & F.col("event_type").isNotNull()
        & F.col("user_id").isNotNull()
        & F.col("product_id").isNotNull()
    )

    # Dedup (conservative). Keep first occurrence.
    # Key chosen to remove exact duplicates without over-collapsing.
    df = df.dropDuplicates(
        ["event_ts", "event_type", "user_id", "product_id", "user_session"]
    )

    # Optional repartition
    if args.repartition and args.repartition > 0:
        df = df.repartition(args.repartition, "year", "month")

    # Write silver fact_events
    (
        df.select(
            "event_ts",
            "event_date",
            "event_type",
            "user_id",
            "user_session",
            "product_id",
            "category_id",
            "category_code",
            "brand",
            "price",
            "year",
            "month",
        )
        .write.mode("overwrite")
        .partitionBy("year", "month")
        .parquet(args.out_silver)
    )

    # Quality report (minimal but useful)
    total = df.count()
    distinct_users = df.select("user_id").distinct().count()
    distinct_products = df.select("product_id").distinct().count()
    et_dist = (
        df.groupBy("event_type")
        .count()
        .orderBy(F.desc("count"))
        .toPandas()
        .to_dict(orient="records")
    )

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows": int(total),
        "distinct_users": int(distinct_users),
        "distinct_products": int(distinct_products),
        "event_type_distribution": et_dist,
        "out_silver": os.path.abspath(args.out_silver),
    }

    os.makedirs("reports", exist_ok=True)
    with open(
        os.path.join("reports", "quality_silver_events.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print(f"[silver.fact_events] rows_written={total:,}")
    print("[quality] reports/quality_silver_events.json written")

    spark.stop()


if __name__ == "__main__":
    main()
