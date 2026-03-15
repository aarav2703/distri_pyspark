import json
import math
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.stats import chi2, norm
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
DUCKDB_INIT = REPO_ROOT / "src" / "sql" / "ddl" / "duckdb_init.sql"
ML_DATASET_PATH = REPO_ROOT / "data" / "gold" / "ml_session_dataset.parquet"
ML_SCORES_PATH = REPO_ROOT / "data" / "gold" / "ml_session_scores.parquet"
EXPERIMENT_ENHANCED_PATH = REPO_ROOT / "data" / "gold" / "experiment_readout_enhanced.parquet"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
METRICS_PATH = REPO_ROOT / "reports" / "ml_metrics.json"
MODEL_DIR = REPO_ROOT / "artifacts" / "models"
DUCKDB_TEMP_DIR = REPO_ROOT / "data" / "duckdb_temp"

FEATURE_COLUMNS = [
    "num_events",
    "n_view",
    "n_cart",
    "session_duration_s",
    "hour",
    "day_of_week",
    "day_of_month",
    "distinct_products_viewed",
    "distinct_categories",
    "avg_view_price",
    "max_view_price",
    "prior_sessions",
    "prior_purchases",
    "prior_conversion_rate",
]


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ML_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)


def init_duckdb(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"SET temp_directory='{DUCKDB_TEMP_DIR.as_posix()}';")
    con.execute("SET threads=4;")
    con.execute("SET preserve_insertion_order=false;")
    with open(DUCKDB_INIT, "r", encoding="utf-8") as f:
        con.execute(f.read())


def build_modeling_dataset(con: duckdb.DuckDBPyConnection) -> None:
    if ML_DATASET_PATH.exists():
        print(f"[ml_dataset] reusing existing dataset at {ML_DATASET_PATH}")
        return

    query = f"""
    COPY (
      WITH session_base AS (
        SELECT
          session_id,
          user_id,
          session_start_ts,
          session_end_ts,
          session_date,
          CAST(num_events AS DOUBLE) AS num_events,
          CAST(n_view AS DOUBLE) AS n_view,
          CAST(n_cart AS DOUBLE) AS n_cart,
          CAST(session_duration_s AS DOUBLE) AS session_duration_s,
          CAST(has_purchase AS INTEGER) AS label,
          EXTRACT('hour' FROM session_start_ts) AS hour,
          EXTRACT('dayofweek' FROM session_start_ts) AS day_of_week,
          EXTRACT('day' FROM session_start_ts) AS day_of_month
        FROM fact_sessions
      ),
      session_event_features AS (
        SELECT
          sha256(
            concat(
              CAST(user_id AS VARCHAR),
              '::',
              COALESCE(user_session, 'NULL')
            )
          ) AS session_id,
          CAST(COUNT(DISTINCT CASE WHEN event_type = 'view' THEN product_id END) AS DOUBLE) AS distinct_products_viewed,
          CAST(
            COUNT(
              DISTINCT CASE
                WHEN category_code IS NOT NULL THEN category_code
                WHEN category_id IS NOT NULL THEN CAST(category_id AS VARCHAR)
                ELSE NULL
              END
            ) AS DOUBLE
          ) AS distinct_categories,
          CAST(AVG(CASE WHEN event_type = 'view' THEN price END) AS DOUBLE) AS avg_view_price,
          CAST(MAX(CASE WHEN event_type = 'view' THEN price END) AS DOUBLE) AS max_view_price
        FROM fact_events
        GROUP BY 1
      ),
      session_history_daily AS (
        SELECT
          user_id,
          session_date,
          COUNT(*) AS sessions_on_day,
          SUM(label) AS purchases_on_day
        FROM session_base
        GROUP BY 1, 2
      ),
      prior_user_features AS (
        SELECT
          user_id,
          session_date,
          COALESCE(
            SUM(sessions_on_day) OVER (
              PARTITION BY user_id
              ORDER BY session_date
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0
          )::DOUBLE AS prior_sessions,
          COALESCE(
            SUM(purchases_on_day) OVER (
              PARTITION BY user_id
              ORDER BY session_date
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0
          )::DOUBLE AS prior_purchases
        FROM session_history_daily
      )
      SELECT
        sb.session_id,
        sb.user_id,
        sb.session_start_ts,
        sb.session_end_ts,
        sb.session_date,
        sb.label,
        sb.num_events,
        sb.n_view,
        sb.n_cart,
        sb.session_duration_s,
        sb.hour::DOUBLE AS hour,
        sb.day_of_week::DOUBLE AS day_of_week,
        sb.day_of_month::DOUBLE AS day_of_month,
        COALESCE(sef.distinct_products_viewed, 0)::DOUBLE AS distinct_products_viewed,
        COALESCE(sef.distinct_categories, 0)::DOUBLE AS distinct_categories,
        COALESCE(sef.avg_view_price, 0)::DOUBLE AS avg_view_price,
        COALESCE(sef.max_view_price, 0)::DOUBLE AS max_view_price,
        COALESCE(puf.prior_sessions, 0)::DOUBLE AS prior_sessions,
        COALESCE(puf.prior_purchases, 0)::DOUBLE AS prior_purchases,
        CASE
          WHEN COALESCE(puf.prior_sessions, 0) > 0
            THEN COALESCE(puf.prior_purchases, 0) / puf.prior_sessions
          ELSE 0
        END::DOUBLE AS prior_conversion_rate
      FROM session_base sb
      LEFT JOIN session_event_features sef
        USING (session_id)
      LEFT JOIN prior_user_features puf
        ON sb.user_id = puf.user_id
       AND sb.session_date = puf.session_date
    ) TO '{ML_DATASET_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)


def print_dataset_profile(con: duckdb.DuckDBPyConnection) -> dict:
    summary = con.execute(
        f"""
        SELECT
          COUNT(*) AS rows,
          AVG(label) AS purchase_rate,
          MIN(session_date) AS min_date,
          MAX(session_date) AS max_date,
          AVG(num_events) AS avg_num_events,
          AVG(n_view) AS avg_n_view,
          AVG(n_cart) AS avg_n_cart,
          AVG(session_duration_s) AS avg_session_duration_s,
          AVG(distinct_products_viewed) AS avg_distinct_products_viewed,
          AVG(distinct_categories) AS avg_distinct_categories,
          AVG(prior_sessions) AS avg_prior_sessions,
          AVG(prior_conversion_rate) AS avg_prior_conversion_rate
        FROM read_parquet('{ML_DATASET_PATH.as_posix()}')
        """
    ).fetchdf()
    schema_df = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{ML_DATASET_PATH.as_posix()}')"
    ).fetchdf()

    print("\n[ml_dataset] schema")
    print(schema_df.to_string(index=False))
    print("\n[ml_dataset] summary")
    print(summary.to_string(index=False))

    return {
        "rows": int(summary.loc[0, "rows"]),
        "purchase_rate": float(summary.loc[0, "purchase_rate"]),
        "min_date": str(summary.loc[0, "min_date"]),
        "max_date": str(summary.loc[0, "max_date"]),
    }


def determine_splits(con: duckdb.DuckDBPyConnection) -> dict:
    dates = [
        row[0]
        for row in con.execute(
            f"""
            SELECT DISTINCT session_date
            FROM read_parquet('{ML_DATASET_PATH.as_posix()}')
            ORDER BY session_date
            """
        ).fetchall()
    ]
    if len(dates) < 3:
        raise ValueError("Need at least 3 distinct session dates for a time-based split.")

    train_idx = max(0, int(len(dates) * 0.7) - 1)
    val_idx = max(train_idx + 1, int(len(dates) * 0.85) - 1)
    val_idx = min(val_idx, len(dates) - 2)

    train_end = dates[train_idx]
    val_end = dates[val_idx]
    test_start = dates[val_idx + 1]

    split_counts = con.execute(
        f"""
        SELECT
          CASE
            WHEN session_date <= DATE '{train_end}' THEN 'train'
            WHEN session_date <= DATE '{val_end}' THEN 'validation'
            ELSE 'test'
          END AS split,
          COUNT(*) AS rows,
          AVG(label) AS purchase_rate
        FROM read_parquet('{ML_DATASET_PATH.as_posix()}')
        GROUP BY 1
        ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END
        """
    ).fetchdf()

    print("\n[splits]")
    print(split_counts.to_string(index=False))

    return {
        "train_end": str(train_end),
        "validation_end": str(val_end),
        "test_start": str(test_start),
        "counts": split_counts.to_dict(orient="records"),
    }


def fetch_split_arrays(
    con: duckdb.DuckDBPyConnection, split_name: str, splits: dict, include_ids: bool = False
):
    if split_name == "train":
        where_sql = f"session_date <= DATE '{splits['train_end']}'"
    elif split_name == "validation":
        where_sql = (
            f"session_date > DATE '{splits['train_end']}' "
            f"AND session_date <= DATE '{splits['validation_end']}'"
        )
    elif split_name == "test":
        where_sql = f"session_date > DATE '{splits['validation_end']}'"
    else:
        raise ValueError(f"Unknown split: {split_name}")

    feature_sql = ", ".join(FEATURE_COLUMNS)
    query = f"""
    SELECT
      {feature_sql},
      label,
      session_id,
      session_date
    FROM read_parquet('{ML_DATASET_PATH.as_posix()}')
    WHERE {where_sql}
    ORDER BY session_start_ts, session_id
    """
    result = con.execute(query).fetchnumpy()

    x = np.column_stack([result[col].astype(np.float32) for col in FEATURE_COLUMNS])
    y = result["label"].astype(np.int8)

    if include_ids:
        ids = pd.DataFrame(
            {
                "session_id": result["session_id"],
                "session_date": result["session_date"],
            }
        )
        return x, y, ids
    return x, y


def sample_for_logistic(x: np.ndarray, y: np.ndarray, max_rows: int = 1_000_000):
    if len(y) <= max_rows:
        return x, y, {"sampled": False, "rows": int(len(y))}

    rng = np.random.default_rng(42)
    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]
    max_pos = min(len(positives), max_rows // 2)
    max_neg = max_rows - max_pos
    pos_idx = rng.choice(positives, size=max_pos, replace=False)
    neg_idx = rng.choice(negatives, size=max_neg, replace=False)
    sample_idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(sample_idx)

    return (
        x[sample_idx],
        y[sample_idx],
        {
            "sampled": True,
            "rows": int(len(sample_idx)),
            "positive_rows": int(max_pos),
            "negative_rows": int(max_neg),
        },
    )


def compute_ranking_metrics(y_true: np.ndarray, y_score: np.ndarray, top_fracs=None) -> dict:
    if top_fracs is None:
        top_fracs = [0.01, 0.05, 0.1]

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    positives = max(int(y_true.sum()), 1)
    baseline_rate = float(y_true.mean())
    metrics = {}

    for frac in top_fracs:
        k = max(1, int(len(y_true) * frac))
        top_y = y_sorted[:k]
        precision_k = float(top_y.mean())
        recall_k = float(top_y.sum() / positives)
        lift_k = float(precision_k / baseline_rate) if baseline_rate > 0 else 0.0
        key = f"top_{int(frac * 100)}pct"
        metrics[key] = {
            "k": int(k),
            "precision": precision_k,
            "recall": recall_k,
            "lift": lift_k,
        }

    return metrics


def build_lift_curve(y_true: np.ndarray, y_score: np.ndarray, bins: int = 20):
    df = pd.DataFrame({"label": y_true, "score": y_score})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["bucket"] = pd.qcut(df.index + 1, q=bins, labels=False, duplicates="drop")
    baseline = df["label"].mean()
    lift_df = (
        df.groupby("bucket", observed=False)
        .agg(
            samples=("label", "size"),
            positives=("label", "sum"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )
    lift_df["precision"] = lift_df["positives"] / lift_df["samples"]
    lift_df["lift"] = np.where(baseline > 0, lift_df["precision"] / baseline, 0.0)
    lift_df["cumulative_fraction"] = (lift_df["samples"].cumsum() / len(df)).astype(float)
    lift_df["cumulative_recall"] = (
        lift_df["positives"].cumsum() / max(int(df["label"].sum()), 1)
    ).astype(float)
    return lift_df


def sample_curve_points(x_values, y_values, max_points: int = 500):
    x_arr = np.asarray(x_values)
    y_arr = np.asarray(y_values)
    if len(x_arr) <= max_points:
        return {"x": x_arr.tolist(), "y": y_arr.tolist()}

    keep = np.linspace(0, len(x_arr) - 1, max_points, dtype=int)
    keep = np.unique(np.concatenate(([0], keep, [len(x_arr) - 1])))
    return {"x": x_arr[keep].tolist(), "y": y_arr[keep].tolist()}


def evaluate_model(name: str, y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_score))
    cm = confusion_matrix(y_true, y_pred).tolist()

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    lift_df = build_lift_curve(y_true, y_score)

    metrics = {
        "name": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,
        "confusion_matrix": cm,
        "ranking": compute_ranking_metrics(y_true, y_score),
        "curve_data": {
            "pr_curve": sample_curve_points(recall_curve, precision_curve),
            "roc_curve": sample_curve_points(fpr, tpr),
            "calibration_curve": {
                "predicted": prob_pred.tolist(),
                "observed": prob_true.tolist(),
            },
            "lift_curve": lift_df.to_dict(orient="records"),
        },
    }

    print(
        f"\n[{name}] PR-AUC={pr_auc:.4f} ROC-AUC={roc_auc:.4f} "
        f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} Brier={brier:.4f}"
    )
    print(f"[{name}] Confusion matrix={cm}")
    for key, value in metrics["ranking"].items():
        print(
            f"[{name}] {key}: precision={value['precision']:.4f} "
            f"recall={value['recall']:.4f} lift={value['lift']:.2f} k={value['k']}"
        )

    return metrics


def save_metric_plots(lightgbm_metrics: dict, feature_importance: pd.DataFrame) -> None:
    pr = lightgbm_metrics["curve_data"]["pr_curve"]
    plt.figure()
    plt.plot(pr["x"], pr["y"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=200)
    plt.close()

    roc = lightgbm_metrics["curve_data"]["roc_curve"]
    plt.figure()
    plt.plot(roc["x"], roc["y"], label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=200)
    plt.close()

    lift_df = pd.DataFrame(lightgbm_metrics["curve_data"]["lift_curve"])
    plt.figure()
    plt.plot(lift_df["cumulative_fraction"], lift_df["lift"])
    plt.xlabel("Cumulative Fraction of Sessions")
    plt.ylabel("Lift")
    plt.title("Lift Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lift_curve.png", dpi=200)
    plt.close()

    calibration = lightgbm_metrics["curve_data"]["calibration_curve"]
    plt.figure()
    plt.plot(calibration["predicted"], calibration["observed"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Purchase Rate")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=200)
    plt.close()

    top_importance = feature_importance.sort_values("importance", ascending=False).head(12)
    plt.figure(figsize=(8, 5))
    plt.barh(top_importance["feature"][::-1], top_importance["importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=200)
    plt.close()


def score_all_sessions(model, output_path: Path) -> dict:
    dataset = ds.dataset(ML_DATASET_PATH, format="parquet")
    writer = None
    total_rows = 0
    score_sum = 0.0

    try:
        for batch in dataset.to_batches(
            columns=FEATURE_COLUMNS + ["session_id", "label"],
            batch_size=250_000,
        ):
            table = pa.Table.from_batches([batch])
            x = np.column_stack(
                [table[col].to_numpy(zero_copy_only=False).astype(np.float32) for col in FEATURE_COLUMNS]
            )
            scores = model.predict_proba(x)[:, 1].astype(np.float32)
            out_table = pa.table(
                {
                    "session_id": table["session_id"],
                    "prediction_score": pa.array(scores),
                    "label": table["label"],
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, out_table.schema, compression="zstd")
            writer.write_table(out_table)
            total_rows += len(scores)
            score_sum += float(scores.sum())
    finally:
        if writer is not None:
            writer.close()

    return {
        "rows": total_rows,
        "avg_prediction_score": score_sum / total_rows if total_rows else 0.0,
    }


def enhance_experiment_readout(con: duckdb.DuckDBPyConnection) -> dict:
    exp_df = con.execute(
        """
        WITH assignment AS (
          SELECT
            user_id,
            CASE WHEN (abs(hash(user_id)) % 2) = 0 THEN 'control' ELSE 'treatment' END AS variant
          FROM (SELECT DISTINCT user_id FROM fact_events)
        ),
        user_purchases AS (
          SELECT
            user_id,
            MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS has_purchase,
            SUM(CASE WHEN event_type = 'purchase' THEN COALESCE(price, 0) ELSE 0 END) AS purchase_revenue
          FROM fact_events
          GROUP BY 1
        ),
        agg AS (
          SELECT
            a.variant,
            COUNT(*) AS assigned_users,
            SUM(COALESCE(up.has_purchase, 0)) AS converters,
            SUM(COALESCE(up.purchase_revenue, 0)) AS revenue
          FROM assignment a
          LEFT JOIN user_purchases up USING (user_id)
          GROUP BY 1
        )
        SELECT * FROM agg ORDER BY variant
        """
    ).fetchdf()

    control = exp_df[exp_df["variant"] == "control"].iloc[0]
    treatment = exp_df[exp_df["variant"] == "treatment"].iloc[0]

    n_c = float(control["assigned_users"])
    n_t = float(treatment["assigned_users"])
    x_c = float(control["converters"])
    x_t = float(treatment["converters"])
    r_c = float(control["revenue"])
    r_t = float(treatment["revenue"])

    p_c = x_c / n_c
    p_t = x_t / n_t
    diff = p_t - p_c
    rel_lift = diff / p_c if p_c > 0 else 0.0
    pooled = (x_c + x_t) / (n_c + n_t)
    se_pooled = math.sqrt(max(pooled * (1 - pooled) * (1 / n_c + 1 / n_t), 1e-12))
    z_value = diff / se_pooled
    p_value = float(2 * norm.sf(abs(z_value)))

    expected = (n_c + n_t) / 2.0
    chi_sq = ((n_c - expected) ** 2 / expected) + ((n_t - expected) ** 2 / expected)
    srm_p_value = float(chi2.sf(chi_sq, df=1))

    se_unpooled = math.sqrt(max((p_c * (1 - p_c) / n_c) + (p_t * (1 - p_t) / n_t), 1e-12))
    ci_low = diff - 1.96 * se_unpooled
    ci_high = diff + 1.96 * se_unpooled

    incremental_conversions = diff * n_t
    control_rpu = r_c / n_c
    treatment_rpu = r_t / n_t
    incremental_revenue = (treatment_rpu - control_rpu) * n_t

    enhanced = pd.DataFrame(
        [
            {
                "experiment_id": "exp_checkout_ui_v1",
                "metric_name": "conversion_rate",
                "control_n": int(n_c),
                "treatment_n": int(n_t),
                "control_converters": int(x_c),
                "treatment_converters": int(x_t),
                "control_value": p_c,
                "treatment_value": p_t,
                "absolute_effect": diff,
                "relative_lift": rel_lift,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "conversion_p_value": p_value,
                "z_statistic": z_value,
                "chi_squared_srm": chi_sq,
                "srm_p_value": srm_p_value,
                "srm_flag": int(srm_p_value < 0.05),
                "incremental_conversions": incremental_conversions,
                "control_revenue": r_c,
                "treatment_revenue": r_t,
                "control_revenue_per_user": control_rpu,
                "treatment_revenue_per_user": treatment_rpu,
                "incremental_revenue": incremental_revenue,
            }
        ]
    )
    enhanced.to_parquet(EXPERIMENT_ENHANCED_PATH, index=False)

    print("\n[experiment_enhanced]")
    print(enhanced.to_string(index=False))

    return enhanced.iloc[0].to_dict()


def main() -> None:
    ensure_dirs()

    con = duckdb.connect(str(DB_PATH))
    init_duckdb(con)

    print("[step] Building downstream modeling dataset")
    build_modeling_dataset(con)
    dataset_profile = print_dataset_profile(con)
    splits = determine_splits(con)

    print("\n[step] Loading split arrays")
    x_train, y_train = fetch_split_arrays(con, "train", splits)
    x_val, y_val = fetch_split_arrays(con, "validation", splits)
    x_test, y_test, _ = fetch_split_arrays(con, "test", splits, include_ids=True)

    print(
        f"[arrays] train={x_train.shape} validation={x_val.shape} test={x_test.shape} "
        f"features={len(FEATURE_COLUMNS)}"
    )

    x_train_lr, y_train_lr, logistic_sampling = sample_for_logistic(x_train, y_train)
    print(f"[logistic] sampling={logistic_sampling}")
    logistic_model = LogisticRegression(
        max_iter=250,
        solver="saga",
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    logistic_model.fit(x_train_lr, y_train_lr)
    logistic_val_scores = logistic_model.predict_proba(x_val)[:, 1]
    logistic_test_scores = logistic_model.predict_proba(x_test)[:, 1]

    scale_pos_weight = max((len(y_train) - y_train.sum()) / max(int(y_train.sum()), 1), 1.0)
    lightgbm_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=100,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    lightgbm_model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    lightgbm_val_scores = lightgbm_model.predict_proba(x_val)[:, 1]
    lightgbm_test_scores = lightgbm_model.predict_proba(x_test)[:, 1]

    print("\n[step] Evaluation")
    logistic_validation_metrics = evaluate_model("logistic_regression_validation", y_val, logistic_val_scores)
    logistic_test_metrics = evaluate_model("logistic_regression_test", y_test, logistic_test_scores)
    lightgbm_validation_metrics = evaluate_model("lightgbm_validation", y_val, lightgbm_val_scores)
    lightgbm_test_metrics = evaluate_model("lightgbm_test", y_test, lightgbm_test_scores)

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": lightgbm_model.feature_importances_.tolist(),
        }
    ).sort_values("importance", ascending=False)
    print("\n[lightgbm] feature importance")
    print(feature_importance.to_string(index=False))
    save_metric_plots(lightgbm_test_metrics, feature_importance)

    print("\n[step] Scoring all sessions")
    score_summary = score_all_sessions(lightgbm_model, ML_SCORES_PATH)
    print(f"[scores] {score_summary}")

    print("\n[step] Enhancing experiment readout")
    experiment_summary = enhance_experiment_readout(con)

    metrics_payload = {
        "assumptions": [
            "Prediction framing is post-session outcome classification on completed sessions.",
            "n_purchase and has_purchase are excluded from model features to avoid direct label leakage.",
            "prior user features are computed from prior session history available before the current session date.",
            "Time-based splits are computed from ordered session dates within the existing November 2019 data.",
            "Logistic regression is trained on a capped sample when the train split exceeds one million rows to keep the baseline tractable on a local machine.",
        ],
        "dataset_profile": dataset_profile,
        "splits": splits,
        "feature_columns": FEATURE_COLUMNS,
        "logistic_sampling": logistic_sampling,
        "models": {
            "logistic_regression": {
                "validation": logistic_validation_metrics,
                "test": logistic_test_metrics,
            },
            "lightgbm": {
                "validation": lightgbm_validation_metrics,
                "test": lightgbm_test_metrics,
                "best_iteration": int(getattr(lightgbm_model, "best_iteration_", 0) or 0),
                "feature_importance": feature_importance.to_dict(orient="records"),
            },
        },
        "score_summary": score_summary,
        "experiment_enhanced": experiment_summary,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    joblib.dump(logistic_model, MODEL_DIR / "session_logistic_regression.joblib")
    joblib.dump(lightgbm_model, MODEL_DIR / "session_lightgbm.joblib")

    con.close()
    print(f"\n[done] metrics={METRICS_PATH} scores={ML_SCORES_PATH} experiment={EXPERIMENT_ENHANCED_PATH}")


if __name__ == "__main__":
    main()
