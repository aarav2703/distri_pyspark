-- Weekly cohort retention based on user's first-ever event week.
-- Output columns:
-- cohort_week, activity_week, week_index, cohort_size, active_users, retention_rate

CREATE OR REPLACE TABLE mart_cohort_weekly_retention AS
WITH user_first AS (
  SELECT
    user_id,
    date_trunc('week', min(event_date))::DATE AS cohort_week
  FROM fact_events
  GROUP BY 1
),
user_weekly_activity AS (
  SELECT
    user_id,
    date_trunc('week', event_date)::DATE AS activity_week
  FROM fact_events
  GROUP BY 1, 2
),
joined AS (
  SELECT
    uf.cohort_week,
    uwa.activity_week,
    date_diff('week', uf.cohort_week, uwa.activity_week) AS week_index,
    uwa.user_id
  FROM user_first uf
  JOIN user_weekly_activity uwa
    ON uf.user_id = uwa.user_id
  WHERE date_diff('week', uf.cohort_week, uwa.activity_week) >= 0
),
cohort_sizes AS (
  SELECT cohort_week, count(*) AS cohort_size
  FROM user_first
  GROUP BY 1
),
ret AS (
  SELECT
    cohort_week,
    activity_week,
    week_index,
    count(DISTINCT user_id) AS active_users
  FROM joined
  GROUP BY 1, 2, 3
)
SELECT
  r.cohort_week,
  r.activity_week,
  r.week_index,
  cs.cohort_size,
  r.active_users,
  (r.active_users::DOUBLE / cs.cohort_size::DOUBLE) AS retention_rate
FROM ret r
JOIN cohort_sizes cs
  USING (cohort_week)
ORDER BY cohort_week, week_index;