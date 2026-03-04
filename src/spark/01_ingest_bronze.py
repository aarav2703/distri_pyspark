import os
import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType


def build_spark(app_name: str) -> SparkSession:
    """
    Windows-safe local SparkSession:
    - Avoids invalid driver RPC URI from hostname (e.g., Legion_7i) by binding to 127.0.0.1
    - Uses local warehouse dir
    - Uses safer output committer settings on Windows
    - Forces RawLocalFileSystem to reduce ChecksumFileSystem/native IO paths
    """
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.warehouse.dir", os.path.abspath("./spark-warehouse"))
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        # --- reduce Windows Hadoop native-IO issues ---
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.speculation", "false")
        # Force RawLocalFileSystem instead of ChecksumFileSystem to avoid NativeIO$Windows.access0 paths
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")
        .config("spark.hadoop.fs.local.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config("spark.hadoop.fs.local.impl.disable.cache", "true")
        # --------------------------------------------
        .getOrCreate()
    )


def event_schema() -> StructType:
    return StructType(
        [
            StructField("event_time", StringType(), True),
            StructField("event_type", StringType(), True),
            StructField("product_id", LongType(), True),
            StructField("category_id", LongType(), True),
            StructField("category_code", StringType(), True),
            StructField("brand", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("user_id", LongType(), True),
            StructField("user_session", StringType(), True),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to raw CSV (or folder of CSVs)"
    )
    parser.add_argument("--out", required=True, help="Output bronze folder (parquet)")
    parser.add_argument(
        "--repartition",
        type=int,
        default=0,
        help="Optional repartition before write (e.g., 16)",
    )
    parser.add_argument(
        "--coalesce",
        type=int,
        default=0,
        help="Optional coalesce before write (e.g., 1). Use only if Windows commits keep failing.",
    )
    args = parser.parse_args()

    spark = build_spark("01_ingest_bronze")

    df = spark.read.option("header", "true").schema(event_schema()).csv(args.input)

    # event_time looks like: "2019-11-01 15:35:08 UTC"
    # Strip trailing " UTC" and parse
    event_time_clean = F.regexp_replace(F.col("event_time"), r"\s+UTC$", "")
    df = df.withColumn(
        "event_ts", F.to_timestamp(event_time_clean, "yyyy-MM-dd HH:mm:ss")
    )

    df = (
        df.withColumn("event_date", F.to_date("event_ts"))
        .withColumn("year", F.year("event_ts"))
        .withColumn("month", F.month("event_ts"))
    )

    # Minimal bronze filters
    df = df.filter(F.col("event_ts").isNotNull())

    # Optional repartition/coalesce controls for Windows stability
    if args.repartition and args.repartition > 0:
        df = df.repartition(args.repartition, "year", "month")

    if args.coalesce and args.coalesce > 0:
        df = df.coalesce(args.coalesce)

    out = args.out

    (
        df.select(
            "event_ts",
            "event_date",
            "event_type",
            "product_id",
            "category_id",
            "category_code",
            "brand",
            "price",
            "user_id",
            "user_session",
            "year",
            "month",
        )
        .write.mode("overwrite")
        .partitionBy("year", "month")
        .parquet(out)
    )

    # Basic run stats
    total = df.count()
    print(f"[bronze] rows_written={total:,}")
    print(f"[bronze] out={os.path.abspath(out)}")

    spark.stop()


if __name__ == "__main__":
    main()
