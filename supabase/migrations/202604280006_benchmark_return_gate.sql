alter table public.recommendation_runs
    add column if not exists min_active_expected_return_vs_benchmark numeric;

alter table public.recommendation_lines
    add column if not exists benchmark_expected_return numeric,
    add column if not exists benchmark_expected_return_margin numeric,
    add column if not exists benchmark_return_gate_passed boolean not null default true;

