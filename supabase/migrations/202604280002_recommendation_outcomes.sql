alter table public.recommendation_lines
    add column if not exists forecast_horizon_days integer,
    add column if not exists forecast_start_date date,
    add column if not exists forecast_end_date date,
    add column if not exists realized_return numeric,
    add column if not exists realized_spy_return numeric,
    add column if not exists realized_active_return numeric,
    add column if not exists forecast_error numeric,
    add column if not exists forecast_hit boolean,
    add column if not exists outcome_status text not null default 'pending';

alter table public.recommendation_lines
    add constraint recommendation_lines_positive_forecast_horizon
    check (forecast_horizon_days is null or forecast_horizon_days > 0) not valid;

alter table public.recommendation_lines
    add constraint recommendation_lines_supported_outcome_status
    check (outcome_status in ('pending', 'realized', 'unavailable')) not valid;

create index if not exists recommendation_lines_forecast_outcome_idx
    on public.recommendation_lines (forecast_start_date, forecast_end_date, outcome_status);
