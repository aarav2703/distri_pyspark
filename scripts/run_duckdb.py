import os
import duckdb

DB_PATH = os.path.join("data", "warehouse.duckdb")


def run_sql_file(con, path: str):
    with open(path, "r", encoding="utf-8") as f:
        con.execute(f.read())


def export_parquet(con, table_name: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    con.execute(
        f"COPY {table_name} TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);"
    )


def main():
    con = duckdb.connect(DB_PATH)

    # init views
    run_sql_file(con, os.path.join("src", "sql", "ddl", "duckdb_init.sql"))

    # sanity check
    n_events = con.execute("SELECT COUNT(*) FROM fact_events").fetchone()[0]
    n_sessions = con.execute("SELECT COUNT(*) FROM fact_sessions").fetchone()[0]
    print("warehouse ready:", {"fact_events": n_events, "fact_sessions": n_sessions})

    # build marts
    run_sql_file(con, os.path.join("src", "sql", "marts", "cohort_retention.sql"))
    run_sql_file(con, os.path.join("src", "sql", "marts", "experiment_readout.sql"))
    run_sql_file(
        con, os.path.join("src", "sql", "marts", "session_funnel_daily.sql")
    )  # NEW

    # export marts to gold/
    export_parquet(
        con,
        "mart_cohort_weekly_retention",
        os.path.join("data", "gold", "mart_cohort_weekly_retention.parquet"),
    )
    export_parquet(
        con,
        "mart_experiment_readout",
        os.path.join("data", "gold", "mart_experiment_readout.parquet"),
    )
    export_parquet(  # NEW
        con,
        "mart_session_funnel_daily",
        os.path.join("data", "gold", "mart_session_funnel_daily.parquet"),
    )

    # quick previews
    print("cohort_retention sample:")
    print(
        con.execute(
            "SELECT * FROM mart_cohort_weekly_retention ORDER BY cohort_week, week_index LIMIT 10"
        )
        .df()
        .to_string(index=False)
    )

    print("\nexperiment_readout:")
    print(
        con.execute(
            "SELECT experiment_id, metric_name, control_value, treatment_value, lift, ci_low, ci_high, srm_flag FROM mart_experiment_readout"
        )
        .df()
        .to_string(index=False)
    )

    print("\nsession_funnel_daily sample:")  # NEW
    print(
        con.execute(
            "SELECT * FROM mart_session_funnel_daily ORDER BY event_date LIMIT 10"
        )
        .df()
        .to_string(index=False)
    )

    con.close()


if __name__ == "__main__":
    main()
