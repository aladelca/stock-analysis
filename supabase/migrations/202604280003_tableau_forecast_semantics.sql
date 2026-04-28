alter table public.recommendation_runs
    add column if not exists expected_return_is_calibrated boolean not null default false;

alter table public.recommendation_lines
    add column if not exists forecast_score numeric,
    add column if not exists calibrated_expected_return numeric,
    add column if not exists expected_return_is_calibrated boolean not null default false;

update public.recommendation_lines
set forecast_score = expected_return
where forecast_score is null
  and expected_return is not null;

alter table public.performance_snapshots
    add column if not exists initial_value numeric,
    add column if not exists invested_capital numeric,
    add column if not exists return_on_invested_capital numeric;

alter table public.performance_snapshots
    add constraint performance_snapshots_nonnegative_initial_value
    check (initial_value is null or initial_value >= 0) not valid;

alter table public.performance_snapshots
    add constraint performance_snapshots_nonnegative_invested_capital
    check (invested_capital is null or invested_capital >= 0) not valid;
