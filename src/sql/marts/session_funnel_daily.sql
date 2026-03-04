-- Session-based funnel by day (based on fact_sessions)
-- Output: event_date, sessions_view, sessions_cart, sessions_purchase, rates

CREATE OR REPLACE TABLE mart_session_funnel_daily AS
WITH s AS (
  SELECT
    session_date AS event_date,
    has_view,
    has_cart,
    has_purchase
  FROM fact_sessions
)
SELECT
  event_date,
  COUNT(*) AS sessions_total,
  SUM(has_view) AS sessions_with_view,
  SUM(has_cart) AS sessions_with_cart,
  SUM(has_purchase) AS sessions_with_purchase,
  CASE WHEN SUM(has_view) > 0 THEN SUM(has_cart)::DOUBLE / SUM(has_view)::DOUBLE ELSE 0 END AS view_to_cart_rate,
CASE WHEN SUM(has_cart) > 0 THEN SUM(CASE WHEN has_cart=1 AND has_purchase=1 THEN 1 ELSE 0 END)::DOUBLE / SUM(has_cart)::DOUBLE ELSE 0 END AS cart_to_purchase_rate,
CASE WHEN SUM(has_view) > 0 THEN SUM(has_purchase)::DOUBLE / SUM(has_view)::DOUBLE ELSE 0 END AS view_to_purchase_rate
FROM s
GROUP BY 1
ORDER BY 1;