import json
import os
from typing import Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def df_profile(df: DataFrame, key_cols=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["rows"] = df.count()
    out["cols"] = len(df.columns)

    # null rates
    null_rates = {}
    for c in df.columns:
        nulls = df.filter(F.col(c).isNull()).count()
        null_rates[c] = nulls / out["rows"] if out["rows"] else 0.0
    out["null_rate"] = null_rates

    # duplicates on key (optional)
    if key_cols:
        dup = df.groupBy(*key_cols).count().filter(F.col("count") > 1).count()
        out["dup_keys"] = dup

    return out


def write_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
