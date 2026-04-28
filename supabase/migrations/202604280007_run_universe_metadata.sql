alter table public.recommendation_runs
    add column if not exists ml_max_assets integer;

alter table public.recommendation_runs
    add constraint recommendation_runs_positive_ml_max_assets
    check (ml_max_assets is null or ml_max_assets > 0) not valid;

