import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DB_PATH = os.path.join("data", "warehouse.duckdb")
OUT_DIR = os.path.join("reports", "figures")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    con = duckdb.connect(DB_PATH)

    # 1) Cohort retention heatmap (cohort_week x week_index)
    cohort = con.execute(
        """
        SELECT cohort_week, week_index, retention_rate
        FROM mart_cohort_weekly_retention
        WHERE week_index BETWEEN 0 AND 8
        ORDER BY cohort_week, week_index
        """
    ).df()

    pivot = cohort.pivot(
        index="cohort_week", columns="week_index", values="retention_rate"
    )

    plt.figure()
    plt.imshow(pivot.fillna(0).values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)

    # Cleaner y-axis labels (date only, no time component)
    y_labels = [pd.to_datetime(d).date().isoformat() for d in pivot.index]
    plt.yticks(range(len(pivot.index)), y_labels)

    plt.xlabel("Week index")
    plt.ylabel("Cohort week")
    plt.title("Weekly Cohort Retention (Week 0–8)")
    plt.colorbar(label="Retention rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cohort_retention_heatmap.png"), dpi=200)
    plt.close()

    # 2) Session funnel rates over time
    funnel = con.execute(
        """
        SELECT event_date,
               view_to_cart_rate,
               cart_to_purchase_rate,
               view_to_purchase_rate
        FROM mart_session_funnel_daily
        ORDER BY event_date
        """
    ).df()

    plt.figure()
    plt.plot(funnel["event_date"], funnel["view_to_cart_rate"], label="view→cart")
    plt.plot(
        funnel["event_date"], funnel["cart_to_purchase_rate"], label="cart→purchase"
    )
    plt.plot(
        funnel["event_date"], funnel["view_to_purchase_rate"], label="view→purchase"
    )
    plt.xlabel("Date")
    plt.ylabel("Rate")
    plt.title("Session Funnel Conversion Rates Over Time")
    plt.legend()

    # Improve x-axis readability
    plt.gca().xaxis.set_major_locator(MaxNLocator(10))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "session_funnel_timeseries.png"), dpi=200)
    plt.close()

    # 3) Experiment readout (lift with CI)
    exp = con.execute(
        """
        SELECT experiment_id, metric_name, control_value, treatment_value, lift, ci_low, ci_high, srm_flag
        FROM mart_experiment_readout
        """
    ).df()

    # single row expected
    row = exp.iloc[0]
    lift = float(row["lift"])
    ci_low = float(row["ci_low"])
    ci_high = float(row["ci_high"])

    plt.figure()
    plt.errorbar(
        [0], [lift], yerr=[[lift - ci_low], [ci_high - lift]], fmt="o", capsize=6
    )
    plt.axhline(0, linewidth=1)
    plt.xticks([0], [f"{row['experiment_id']} / {row['metric_name']}"])
    plt.ylabel("Lift (treatment - control)")
    plt.title(f"Experiment Lift with 95% CI (SRM flag={int(row['srm_flag'])})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "experiment_readout.png"), dpi=200)
    plt.close()

    con.close()
    print("Wrote figures to:", OUT_DIR)


if __name__ == "__main__":
    main()
