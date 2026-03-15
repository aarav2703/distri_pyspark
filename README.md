# Distributed Product Analytics & Experimentation Platform (E-Commerce Events)

This project builds an end-to-end product analytics pipeline on large-scale e-commerce event logs. The goal was to demonstrate the intersection of:

- **PySpark data engineering** (ETL + sessionization + partitioned lakehouse tables)
- **Warehouse analytics in SQL** (DuckDB marts for funnels and cohorts)
- **Experimentation analysis** (A/B assignment, SRM checks, lift + confidence intervals)
- **Analytics outputs** that look like what product and growth teams actually use

I picked this dataset because it forces the same real tradeoffs you see in production: noisy event logs, huge volume, session reconstruction, and metric definitions that have to be consistent across tables.

---

## Architecture

```mermaid
flowchart LR
A[Raw Event Logs\nCSV / GZ] --> B[Bronze Parquet\nImmutable storage]
B --> C[Silver ETL in PySpark\nSchema + cleaning]
C --> D[Sessionization\nfact_sessions]
C --> E[Gold aggregates\nuser_daily / product_daily / funnel_daily]
D --> F[DuckDB Warehouse\nViews over Parquet]
E --> F
F --> G1[mart_session_funnel_daily]
F --> G2[mart_cohort_weekly_retention]
F --> G3[mart_experiment_readout]
G1 --> H[Figures / Reporting]
G2 --> H
G3 --> H
```

## Dataset
Source: multi-category online store event logs (Nov 2019 used in this run).
Each row represents an event such as view, cart, or purchase with identifiers for the user, product, and session.

## Pipeline Overview

### 1) Bronze: raw → partitioned Parquet
Why: keeping a bronze layer makes the pipeline repeatable. If downstream logic changes, silver/gold can be rebuilt without touching the original input.

Input: `data/raw/2019-Nov.csv`
Output: `data/bronze/events/year=2019/month=11/...`

### 2) Silver: cleaned fact table (fact_events)
Why: analytics tables are only as good as the raw event hygiene. Silver is where schemas become stable, event types become canonical, and obvious corruption gets filtered.

Output: `data/silver/fact_events`

### 3) Silver: session reconstruction (fact_sessions)
Why: product funnels and engagement are often more stable and interpretable at the session level than the event level.

Session id: deterministic hash of (user_id, user_session)
Output: `data/silver/fact_sessions`

### 4) Gold: daily aggregates
Why: pushing heavy aggregation into Spark keeps the SQL layer lightweight and fast.

`data/gold/user_daily`
`data/gold/product_daily`
`data/gold/funnel_daily`

### 5) DuckDB warehouse + SQL marts
Why DuckDB: it gives a “warehouse feel” locally (SQL marts, reproducible queries) without needing a full cloud warehouse to demonstrate the modeling skill.
Views over Parquet:

* fact_events
* fact_sessions
* user_daily
* product_daily
* funnel_daily

Marts:

* mart_session_funnel_daily
* mart_cohort_weekly_retention
* mart_experiment_readout

### 6) Reporting / figures
Three outputs that summarize the product analytics story:

* cohort retention heatmap
* session funnel rates over time
* experiment lift with 95% CI (with SRM flag)

Additional report figures in `reports/figures/`:

* `cohort retention report.png`
* `funnel trends.png`

## Metrics (what the pipeline actually computes)

### `fact_sessions` (session-level)
- duration: `session_duration_s`
- volume: `num_events`
- stage counts: `n_view`, `n_cart`, `n_purchase`
- funnel flags: `has_view`, `has_cart`, `has_purchase`

### `mart_session_funnel_daily` (session funnel)
- counts: `sessions_total`, `sessions_with_view`, `sessions_with_cart`, `sessions_with_purchase`
- rates: `view_to_cart_rate`, `cart_to_purchase_rate`, `view_to_purchase_rate`

### `mart_cohort_weekly_retention` (cohorts)
- `cohort_size`, `active_users`, `retention_rate` by `week_index`

### `mart_experiment_readout` (A/B readout)
- conversion: `control_value`, `treatment_value`, `lift`, `ci_low`, `ci_high`
- validity: `srm_flag`, plus `control_n` / `treatment_n`
- revenue impact: `control_revenue_per_user`, `treatment_revenue_per_user`, `lift_revenue_per_user`

## Outputs (from this run)
Scale:

* 67.4M cleaned events (fact_events)
* 13.8M sessions (fact_sessions)
* Weekly cohort retention (Week 0–8)
* Session funnel conversion rates over time
* Experiment lift with 95% CI (+ SRM check)

## How to run (Windows / local)
Environment
This repo assumes PySpark is working locally (Java + Hadoop configured) and that you're running inside your prod-analytics conda environment.
I set these in PowerShell to avoid PySpark defaulting to a missing python3:

```powershell
$env:PYSPARK_PYTHON = (Get-Command python).Source
$env:PYSPARK_DRIVER_PYTHON = (Get-Command python).Source
```

### 1) Build lakehouse layers (PySpark)
```powershell
python src\spark\01_ingest_bronze.py --input data\raw\2019-Nov.csv --out data\bronze\events --repartition 8
python src\spark\02_clean_silver.py  --in_bronze data\bronze\events --out_silver data\silver\fact_events --repartition 8
python src\spark\03_sessions.py      --in_events data\silver\fact_events --out_sessions data\silver\fact_sessions --repartition 8
python src\spark\04_features_user_product.py --in_events data\silver\fact_events --out_gold data\gold --repartition 8
```

### 2) Build SQL marts (DuckDB)
```powershell
python scripts\run_duckdb.py
```

### 3) Generate figures
```powershell
python scripts\make_figures.py
```

### 4) Train downstream ML model + enhanced experiment readout
This project includes post-session outcome classification, predicting whether a completed session contains a purchase.

The downstream modeling layer reads the existing Parquet outputs and DuckDB marts without rerunning Spark.

```powershell
python train_model.py
```

Artifacts written by the downstream ML layer:

* `data/gold/ml_session_dataset.parquet`
* `data/gold/ml_session_scores.parquet`
* `data/gold/experiment_readout_enhanced.parquet`
* `reports/ml_metrics.json`
* `reports/figures/pr_curve.png`
* `reports/figures/roc_curve.png`
* `reports/figures/lift_curve.png`
* `reports/figures/calibration_curve.png`
* `reports/figures/feature_importance.png`
* `reports/figures/cohort retention report.png`
* `reports/figures/funnel trends.png`

### 5) Launch the dashboard
```powershell
streamlit run dashboard/app.py
```

The dashboard includes:

* overview cards for sessions, conversion, retention, and experiment results
* session funnel trends
* cohort retention heatmap
* experiment analysis with enhanced statistics
* ML performance views for PR, ROC, lift, and calibration
* explainability views for feature importance and score distributions

## Predictive modeling
The downstream ML task is a session purchase outcome classifier built from `fact_sessions` plus additional downstream features from `fact_events` and prior session history.

Label:

* `has_purchase`

Core features:

* `num_events`
* `n_view`
* `n_cart`
* `session_duration_s`
* session-start calendar features (`hour`, `day_of_week`, `day_of_month`)
* distinct products viewed, category diversity, and view-price aggregates
* prior user session count, prior purchases, and prior conversion rate

Excluded to avoid direct leakage as model inputs:

* `has_purchase`
* `n_purchase`
* identifiers such as `session_id` and `user_id`

Models:

* Logistic Regression baseline
* LightGBM main model

Validation:

* time-based split over ordered session dates

## Evaluation metrics
The downstream classifier reports:

* PR-AUC
* ROC-AUC
* Precision / Recall / F1
* Recall@k
* Precision@k
* Lift@k
* Brier score
* calibration curve
* confusion matrix
* feature importance

Metrics and curve data are stored in `reports/ml_metrics.json` for reuse in the dashboard.

## Experiment statistics
The enhanced downstream experiment readout extends the original mart with:

* chi-squared SRM test
* two-proportion z-test p-value for conversion
* absolute effect size
* relative lift
* incremental conversions
* incremental revenue

Enhanced output:

* `data/gold/experiment_readout_enhanced.parquet`

## Dashboard usage
The Streamlit app reads directly from DuckDB and Parquet outputs. Run:

```powershell
streamlit run dashboard/app.py
```

If the ML artifacts are missing, run:

```powershell
python train_model.py
```

## Notes on design choices (brief)
The bronze/silver/gold layout makes debugging and iteration easier. It also shows a lightweight version of production-style data discipline in a local project.
Session-based funnels are used because event-level funnels can be misleading when telemetry is incomplete. Sessions provide a cleaner denominator for conversion analysis.
The experiment is simulated, and the downstream enhanced readout adds a proper SRM significance test, conversion p-value, and absolute effect reporting on top of the existing event-level data.

## Data quality checks
Implemented checks that can be verified from the repository:

* event schema is enforced during bronze ingestion
* rows with null parsed timestamps are dropped in bronze
* silver events keep only canonical event types: `view`, `cart`, `purchase`
* silver events require non-null `event_ts`, `event_type`, `user_id`, and `product_id`
* conservative event deduplication is applied on `(event_ts, event_type, user_id, product_id, user_session)`
* quality reports are written to `reports/quality_silver_events.json` and `reports/quality_silver_sessions.json`

## Common pitfalls I guarded against
* **Event-Level vs. Session-Level Funnels:** Raw event-level funnels can hallucinate drop-offs when telemetry is incomplete. Session-level aggregation provides a more stable denominator for conversion rates.
* **Sample Ratio Mismatch (SRM):** The original SQL mart includes a simple split-balance flag, and the downstream enhanced experiment readout adds an explicit chi-squared SRM test with a reported p-value.

