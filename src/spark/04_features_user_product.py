import os
import json
import argparse
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark(app_name: str) -> SparkSession:
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
    ap.add_argument("--in_events", required=True)
    ap.add_argument(
        "--out_gold", required=True, help="Gold output root, e.g. data/gold"
    )
    ap.add_argument("--repartition", type=int, default=8)
    args = ap.parse_args()

    spark = build_spark("04_features_user_product")

    events = spark.read.parquet(args.in_events)

    # Ensure we have these columns
    # event_date, event_type, user_id, product_id, price
    events = events.filter(
        F.col("event_date").isNotNull()
        & F.col("event_type").isNotNull()
        & F.col("user_id").isNotNull()
        & F.col("product_id").isNotNull()
    )

    # Common metrics helpers
    is_view = F.when(F.col("event_type") == "view", 1).otherwise(0)
    is_cart = F.when(F.col("event_type") == "cart", 1).otherwise(0)
    is_purchase = F.when(F.col("event_type") == "purchase", 1).otherwise(0)
    purchase_revenue = F.when(
        F.col("event_type") == "purchase", F.coalesce(F.col("price"), F.lit(0.0))
    ).otherwise(F.lit(0.0))

    # -------------------------
    # 1) Daily product metrics
    # -------------------------
    product_daily = (
        events.groupBy("event_date", "product_id")
        .agg(
            F.sum(is_view).alias("views"),
            F.sum(is_cart).alias("carts"),
            F.sum(is_purchase).alias("purchases"),
            F.sum(purchase_revenue).alias("revenue"),
            F.countDistinct("user_id").alias("unique_users"),
        )
        .withColumn(
            "view_to_cart_rate",
            F.when(F.col("views") > 0, F.col("carts") / F.col("views")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn(
            "cart_to_purchase_rate",
            F.when(F.col("carts") > 0, F.col("purchases") / F.col("carts")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn(
            "view_to_purchase_rate",
            F.when(F.col("views") > 0, F.col("purchases") / F.col("views")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn("year", F.year("event_date"))
        .withColumn("month", F.month("event_date"))
    )

    # ----------------------
    # 2) Daily user metrics
    # ----------------------
    user_daily = (
        events.groupBy("event_date", "user_id")
        .agg(
            F.sum(is_view).alias("views"),
            F.sum(is_cart).alias("carts"),
            F.sum(is_purchase).alias("purchases"),
            F.sum(purchase_revenue).alias("revenue"),
            F.countDistinct("product_id").alias("unique_products"),
        )
        .withColumn(
            "cart_rate",
            F.when(F.col("views") > 0, F.col("carts") / F.col("views")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn(
            "purchase_rate",
            F.when(F.col("views") > 0, F.col("purchases") / F.col("views")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn("year", F.year("event_date"))
        .withColumn("month", F.month("event_date"))
    )

    # ----------------------
    # 3) Daily funnel (site-wide)
    # ----------------------
    funnel_daily = (
        events.groupBy("event_date")
        .agg(
            F.sum(is_view).alias("views"),
            F.sum(is_cart).alias("carts"),
            F.sum(is_purchase).alias("purchases"),
            F.countDistinct(
                F.when(F.col("event_type") == "view", F.col("user_id"))
            ).alias("view_users"),
            F.countDistinct(
                F.when(F.col("event_type") == "cart", F.col("user_id"))
            ).alias("cart_users"),
            F.countDistinct(
                F.when(F.col("event_type") == "purchase", F.col("user_id"))
            ).alias("purchase_users"),
        )
        .withColumn(
            "view_to_cart_rate",
            F.when(F.col("views") > 0, F.col("carts") / F.col("views")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn(
            "cart_to_purchase_rate",
            F.when(F.col("carts") > 0, F.col("purchases") / F.col("carts")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn(
            "view_to_purchase_rate",
            F.when(F.col("views") > 0, F.col("purchases") / F.col("views")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn("year", F.year("event_date"))
        .withColumn("month", F.month("event_date"))
    )

    out_root = args.out_gold
    os.makedirs(out_root, exist_ok=True)

    # Write partitioned outputs
    if args.repartition and args.repartition > 0:
        product_daily = product_daily.repartition(args.repartition, "year", "month")
        user_daily = user_daily.repartition(args.repartition, "year", "month")
        funnel_daily = funnel_daily.repartition(args.repartition, "year", "month")

    product_path = os.path.join(out_root, "product_daily")
    user_path = os.path.join(out_root, "user_daily")
    funnel_path = os.path.join(out_root, "funnel_daily")

    product_daily.write.mode("overwrite").partitionBy("year", "month").parquet(
        product_path
    )
    user_daily.write.mode("overwrite").partitionBy("year", "month").parquet(user_path)
    funnel_daily.write.mode("overwrite").partitionBy("year", "month").parquet(
        funnel_path
    )

    # Report (avoid df.count() on huge tables; sample via approx stats)
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "out_gold": os.path.abspath(out_root),
        "tables": {
            "product_daily": product_path,
            "user_daily": user_path,
            "funnel_daily": funnel_path,
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open(
        os.path.join("reports", "quality_gold_features.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("[gold] wrote: product_daily, user_daily, funnel_daily")
    print("[quality] reports/quality_gold_features.json written")

    spark.stop()


if __name__ == "__main__":
    main()
