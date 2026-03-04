-- Deterministic A/B experiment simulation using user_id hash mod 2.
-- Metrics:
-- 1) conversion_rate = users_with_purchase / assigned_users
-- 2) revenue_per_user = total_purchase_revenue / assigned_users
-- Includes SRM check and 95% CI for conversion lift.

CREATE OR REPLACE TABLE mart_experiment_readout AS
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
    SUM(CASE WHEN event_type = 'purchase' THEN COALESCE(price,0) ELSE 0 END) AS purchase_revenue
  FROM fact_events
  GROUP BY 1
),
agg AS (
  SELECT
    a.variant,
    COUNT(*) AS assigned_users,
    SUM(up.has_purchase) AS converters,
    SUM(up.purchase_revenue) AS revenue
  FROM assignment a
  LEFT JOIN user_purchases up
    USING (user_id)
  GROUP BY 1
),
paired AS (
  SELECT
    MAX(CASE WHEN variant='control' THEN assigned_users END) AS n_c,
    MAX(CASE WHEN variant='treatment' THEN assigned_users END) AS n_t,
    MAX(CASE WHEN variant='control' THEN converters END) AS x_c,
    MAX(CASE WHEN variant='treatment' THEN converters END) AS x_t,
    MAX(CASE WHEN variant='control' THEN revenue END) AS r_c,
    MAX(CASE WHEN variant='treatment' THEN revenue END) AS r_t
  FROM agg
),
stats AS (
  SELECT
    n_c, n_t, x_c, x_t, r_c, r_t,
    (x_c::DOUBLE / n_c::DOUBLE) AS p_c,
    (x_t::DOUBLE / n_t::DOUBLE) AS p_t,
    (r_c::DOUBLE / n_c::DOUBLE) AS rpu_c,
    (r_t::DOUBLE / n_t::DOUBLE) AS rpu_t
  FROM paired
),
ci AS (
  SELECT
    *,
    (p_t - p_c) AS lift_conv,
    -- standard error for diff in proportions
    sqrt( (p_c*(1-p_c))/n_c + (p_t*(1-p_t))/n_t ) AS se_diff,
    (p_t - p_c) - 1.96*sqrt( (p_c*(1-p_c))/n_c + (p_t*(1-p_t))/n_t ) AS ci_low,
    (p_t - p_c) + 1.96*sqrt( (p_c*(1-p_c))/n_c + (p_t*(1-p_t))/n_t ) AS ci_high,
    (rpu_t - rpu_c) AS lift_rpu
  FROM stats
),
srm AS (
  SELECT
    *,
    -- SRM: expected 50/50 split. Flag if deviation > 1%.
    CASE WHEN abs((n_t::DOUBLE/(n_c+n_t)) - 0.5) > 0.01 THEN 1 ELSE 0 END AS srm_flag
  FROM ci
)
SELECT
  'exp_checkout_ui_v1' AS experiment_id,

  -- Conversion metric
  'conversion_rate' AS metric_name,
  p_c AS control_value,
  p_t AS treatment_value,
  lift_conv AS lift,
  ci_low,
  ci_high,
  NULL::DOUBLE AS p_value,     -- optional later if you want z-test p-value
  srm_flag,

  -- Extra fields for debugging/reporting
  n_c AS control_n,
  n_t AS treatment_n,
  x_c AS control_converters,
  x_t AS treatment_converters,
  rpu_c AS control_revenue_per_user,
  rpu_t AS treatment_revenue_per_user,
  lift_rpu AS lift_revenue_per_user
FROM srm;