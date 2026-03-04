-- DuckDB init: create views over Parquet layers

-- SILVER
CREATE OR REPLACE VIEW fact_events AS
SELECT * FROM read_parquet('data/silver/fact_events/**/*.parquet');

CREATE OR REPLACE VIEW fact_sessions AS
SELECT * FROM read_parquet('data/silver/fact_sessions/**/*.parquet');

-- GOLD
CREATE OR REPLACE VIEW funnel_daily AS
SELECT * FROM read_parquet('data/gold/funnel_daily/**/*.parquet');

CREATE OR REPLACE VIEW user_daily AS
SELECT * FROM read_parquet('data/gold/user_daily/**/*.parquet');

CREATE OR REPLACE VIEW product_daily AS
SELECT * FROM read_parquet('data/gold/product_daily/**/*.parquet');