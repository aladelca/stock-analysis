alter table public.recommendation_lines
    add column if not exists executable_target_weight numeric,
    add column if not exists executable_target_market_value numeric;

create index if not exists recommendation_runs_account_data_status_idx
    on public.recommendation_runs (account_id, data_as_of_date, status, created_at desc);
