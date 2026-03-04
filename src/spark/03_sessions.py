import os
import json
import argparse
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark(app_name: str) -> SparkSession:
    # Match the stable approach you used in 02
    return (
        SparkSession.builder.appName(app_name)
        .master("local[4]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.warehouse.dir", os.path.abspath("./spark-warehouse"))
        .getOrCreate()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_events", required=True, help="Path to silver fact_events parquet root"
    )
    ap.add_argument(
        "--out_sessions",
        required=True,
        help="Path to silver fact_sessions parquet root",
    )
    ap.add_argument(
        "--repartition", type=int, default=0, help="Optional repartition before write"
    )
    args = ap.parse_args()

    spark = build_spark("03_build_sessions")

    events = spark.read.parquet(args.in_events)

    # Stable session id: hash(user_id + user_session)
    # user_session can be null in some rows; keep it explicit to avoid collisions
    events = events.withColumn(
        "session_id",
        F.sha2(
            F.concat_ws(
                "::",
                F.col("user_id").cast("string"),
                F.coalesce(F.col("user_session"), F.lit("NULL")),
            ),
            256,
        ),
    )

    # Session aggregates
    sess = (
        events.groupBy("session_id", "user_id")
        .agg(
            F.min("event_ts").alias("session_start_ts"),
            F.max("event_ts").alias("session_end_ts"),
            F.count(F.lit(1)).alias("num_events"),
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias(
                "n_view"
            ),
            F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias(
                "n_cart"
            ),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias(
                "n_purchase"
            ),
        )
        .withColumn(
            "session_duration_s",
            F.col("session_end_ts").cast("long")
            - F.col("session_start_ts").cast("long"),
        )
        .withColumn("has_view", (F.col("n_view") > 0).cast("int"))
        .withColumn("has_cart", (F.col("n_cart") > 0).cast("int"))
        .withColumn("has_purchase", (F.col("n_purchase") > 0).cast("int"))
        .withColumn("session_date", F.to_date("session_start_ts"))
        .withColumn("year", F.year("session_start_ts"))
        .withColumn("month", F.month("session_start_ts"))
    )

    if args.repartition and args.repartition > 0:
        sess = sess.repartition(args.repartition, "year", "month")

    (
        sess.select(
            "session_id",
            "user_id",
            "session_start_ts",
            "session_end_ts",
            "session_duration_s",
            "num_events",
            "n_view",
            "n_cart",
            "n_purchase",
            "has_view",
            "has_cart",
            "has_purchase",
            "session_date",
            "year",
            "month",
        )
        .write.mode("overwrite")
        .partitionBy("year", "month")
        .parquet(args.out_sessions)
    )

    # Small quality report
    total_sessions = sess.count()
    distinct_users = sess.select("user_id").distinct().count()
    funnel_dist = (
        sess.select(
            (F.col("has_view") == 1).alias("view"),
            (F.col("has_cart") == 1).alias("cart"),
            (F.col("has_purchase") == 1).alias("purchase"),
        )
        .groupBy("view", "cart", "purchase")
        .count()
        .orderBy(F.desc("count"))
        .toPandas()
        .to_dict(orient="records")
    )

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sessions": int(total_sessions),
        "distinct_users": int(distinct_users),
        "funnel_flag_distribution": funnel_dist[:20],
        "out_sessions": os.path.abspath(args.out_sessions),
    }

    os.makedirs("reports", exist_ok=True)
    with open(
        os.path.join("reports", "quality_silver_sessions.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print(f"[silver.fact_sessions] sessions_written={total_sessions:,}")
    print("[quality] reports/quality_silver_sessions.json written")

    spark.stop()


if __name__ == "__main__":
    main()
