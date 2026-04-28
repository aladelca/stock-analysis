alter table public.recommendation_runs
    add column if not exists optimizer_return_unit text,
    add column if not exists calibration_enabled boolean not null default false,
    add column if not exists calibration_method text,
    add column if not exists calibration_target text,
    add column if not exists calibration_model_version text,
    add column if not exists calibration_status text,
    add column if not exists calibration_trained_through_date date,
    add column if not exists calibration_observations integer,
    add column if not exists calibration_mae numeric,
    add column if not exists calibration_rmse numeric,
    add column if not exists calibration_rank_ic numeric;

alter table public.recommendation_runs
    add constraint recommendation_runs_nonnegative_calibration_observations
    check (calibration_observations is null or calibration_observations >= 0) not valid;

create index if not exists recommendation_runs_calibration_status_idx
    on public.recommendation_runs (calibration_status, calibration_trained_through_date);
