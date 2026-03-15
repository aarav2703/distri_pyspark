import json
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
ML_METRICS_PATH = REPO_ROOT / "reports" / "ml_metrics.json"
ML_SCORES_PATH = REPO_ROOT / "data" / "gold" / "ml_session_scores.parquet"
EXPERIMENT_ENHANCED_PATH = REPO_ROOT / "data" / "gold" / "experiment_readout_enhanced.parquet"


st.set_page_config(page_title="Session Analytics & ML Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_metrics():
    if not ML_METRICS_PATH.exists():
        return None
    with open(ML_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)


@st.cache_data(show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    con = get_connection()
    return con.execute(sql).fetchdf()


@st.cache_data(show_spinner=False)
def load_scores(sample_rows: int = 500_000) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM read_parquet('{ML_SCORES_PATH.as_posix()}')
    USING SAMPLE {sample_rows} ROWS
    """
    return duckdb.sql(query).df()


def render_overview(metrics):
    summary = run_query(
        """
        SELECT
          (SELECT COUNT(*) FROM fact_sessions) AS total_sessions,
          (SELECT AVG(has_purchase) FROM fact_sessions) AS conversion_rate,
          (SELECT AVG(retention_rate) FROM mart_cohort_weekly_retention WHERE week_index = 1) AS week1_retention,
          (SELECT lift FROM mart_experiment_readout) AS experiment_lift
        """
    ).iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions", f"{int(summary['total_sessions']):,}")
    col2.metric("Conversion Rate", f"{summary['conversion_rate']:.2%}")
    col3.metric("Week 1 Retention", f"{summary['week1_retention']:.2%}")
    col4.metric("Experiment Lift", f"{summary['experiment_lift']:.3%}")

    if metrics:
        ml = metrics["models"]["lightgbm"]["test"]
        st.subheader("Model Snapshot")
        cols = st.columns(4)
        cols[0].metric("PR-AUC", f"{ml['pr_auc']:.4f}")
        cols[1].metric("ROC-AUC", f"{ml['roc_auc']:.4f}")
        cols[2].metric("Recall@10%", f"{ml['ranking']['top_10pct']['recall']:.2%}")
        cols[3].metric("Lift@10%", f"{ml['ranking']['top_10pct']['lift']:.2f}x")

    if EXPERIMENT_ENHANCED_PATH.exists():
        st.subheader("Enhanced Experiment Summary")
        exp = pd.read_parquet(EXPERIMENT_ENHANCED_PATH)
        st.dataframe(exp, use_container_width=True)


def render_funnel_trends():
    funnel = run_query(
        """
        SELECT *
        FROM mart_session_funnel_daily
        ORDER BY event_date
        """
    )
    fig = go.Figure()
    for col in ["view_to_cart_rate", "cart_to_purchase_rate", "view_to_purchase_rate"]:
        fig.add_trace(go.Scatter(x=funnel["event_date"], y=funnel[col], mode="lines", name=col))
    fig.update_layout(title="Session Funnel Rates", xaxis_title="Date", yaxis_title="Rate")
    st.plotly_chart(fig, use_container_width=True)

    fig_counts = go.Figure()
    for col in ["sessions_total", "sessions_with_view", "sessions_with_cart", "sessions_with_purchase"]:
        fig_counts.add_trace(go.Scatter(x=funnel["event_date"], y=funnel[col], mode="lines", name=col))
    fig_counts.update_layout(title="Session Funnel Counts", xaxis_title="Date", yaxis_title="Sessions")
    st.plotly_chart(fig_counts, use_container_width=True)


def render_cohorts():
    cohort = run_query(
        """
        SELECT cohort_week, week_index, retention_rate
        FROM mart_cohort_weekly_retention
        ORDER BY cohort_week, week_index
        """
    )
    pivot = cohort.pivot(index="cohort_week", columns="week_index", values="retention_rate")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Week Index", y="Cohort Week", color="Retention"),
        title="Weekly Cohort Retention",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_experiment():
    base = run_query("SELECT * FROM mart_experiment_readout")
    st.subheader("Current Mart")
    st.dataframe(base, use_container_width=True)

    if EXPERIMENT_ENHANCED_PATH.exists():
        enhanced = pd.read_parquet(EXPERIMENT_ENHANCED_PATH)
        st.subheader("Enhanced Readout")
        st.dataframe(enhanced, use_container_width=True)

        row = enhanced.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Absolute Effect", f"{row['absolute_effect']:.3%}")
        cols[1].metric("Relative Lift", f"{row['relative_lift']:.2%}")
        cols[2].metric("Conv p-value", f"{row['conversion_p_value']:.4g}")
        cols[3].metric("SRM p-value", f"{row['srm_p_value']:.4g}")
        cols[4].metric("Incremental Revenue", f"{row['incremental_revenue']:.2f}")


def render_ml_performance(metrics):
    if not metrics:
        st.warning("Run `python train_model.py` to generate ML metrics.")
        return

    lightgbm = metrics["models"]["lightgbm"]["test"]

    pr = pd.DataFrame(
        {
            "recall": lightgbm["curve_data"]["pr_curve"]["x"],
            "precision": lightgbm["curve_data"]["pr_curve"]["y"],
        }
    )
    fig_pr = px.line(pr, x="recall", y="precision", title="Precision-Recall Curve")
    st.plotly_chart(fig_pr, use_container_width=True)

    roc = pd.DataFrame(
        {
            "fpr": lightgbm["curve_data"]["roc_curve"]["x"],
            "tpr": lightgbm["curve_data"]["roc_curve"]["y"],
        }
    )
    fig_roc = px.line(roc, x="fpr", y="tpr", title="ROC Curve")
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig_roc, use_container_width=True)

    lift = pd.DataFrame(lightgbm["curve_data"]["lift_curve"])
    fig_lift = px.line(
        lift,
        x="cumulative_fraction",
        y="lift",
        title="Lift Curve",
        labels={"cumulative_fraction": "Cumulative Fraction of Sessions"},
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    calibration = pd.DataFrame(lightgbm["curve_data"]["calibration_curve"])
    fig_cal = px.line(
        calibration,
        x="predicted",
        y="observed",
        markers=True,
        title="Calibration Curve",
        labels={"predicted": "Predicted Probability", "observed": "Observed Purchase Rate"},
    )
    fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig_cal, use_container_width=True)

    cols = st.columns(6)
    cols[0].metric("PR-AUC", f"{lightgbm['pr_auc']:.4f}")
    cols[1].metric("ROC-AUC", f"{lightgbm['roc_auc']:.4f}")
    cols[2].metric("Precision", f"{lightgbm['precision']:.4f}")
    cols[3].metric("Recall", f"{lightgbm['recall']:.4f}")
    cols[4].metric("F1", f"{lightgbm['f1']:.4f}")
    cols[5].metric("Brier", f"{lightgbm['brier_score']:.4f}")


def render_explainability(metrics):
    if not metrics or not ML_SCORES_PATH.exists():
        st.warning("Run `python train_model.py` to generate explainability artifacts.")
        return

    importance = pd.DataFrame(metrics["models"]["lightgbm"]["feature_importance"]).sort_values(
        "importance", ascending=False
    )
    fig_imp = px.bar(
        importance.head(15),
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    scores = load_scores()
    scores["label_name"] = scores["label"].map({0: "No Purchase", 1: "Purchase"})
    fig_scores = px.histogram(
        scores,
        x="prediction_score",
        color="label_name",
        nbins=50,
        barmode="overlay",
        histnorm="probability density",
        title="Score Distribution by Label",
    )
    st.plotly_chart(fig_scores, use_container_width=True)


def main():
    st.title("Distributed Product Analytics, Experimentation, and ML")
    metrics = load_metrics()

    page = st.sidebar.radio(
        "Pages",
        [
            "Overview",
            "Funnel Trends",
            "Cohort Retention",
            "Experiment Analysis",
            "ML Model Performance",
            "Explainability",
        ],
    )

    if page == "Overview":
        render_overview(metrics)
    elif page == "Funnel Trends":
        render_funnel_trends()
    elif page == "Cohort Retention":
        render_cohorts()
    elif page == "Experiment Analysis":
        render_experiment()
    elif page == "ML Model Performance":
        render_ml_performance(metrics)
    else:
        render_explainability(metrics)


if __name__ == "__main__":
    main()
