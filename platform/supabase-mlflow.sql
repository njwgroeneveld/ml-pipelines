-- Run ONCE in the Supabase SQL editor of the trading project.
-- Replace CHANGE-ME with a strong password before running, and do not save the edited text anywhere.
-- MLflow gets its own schema and its own login, without access to the trading tables.
create schema if not exists mlflow;
create role mlflow_user login password 'CHANGE-ME';
grant usage, create on schema mlflow to mlflow_user;
-- MLflow then creates its tables in schema mlflow, not in public.
alter role mlflow_user set search_path = mlflow;
