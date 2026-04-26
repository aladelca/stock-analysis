create extension if not exists pgcrypto;

create table if not exists public.accounts (
    id uuid primary key default gen_random_uuid(),
    slug text not null unique,
    display_name text not null,
    base_currency text not null default 'USD',
    benchmark_ticker text not null default 'SPY',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint accounts_slug_not_blank check (length(btrim(slug)) > 0),
    constraint accounts_currency_upper check (base_currency = upper(base_currency))
);

create table if not exists public.portfolio_snapshots (
    id uuid primary key default gen_random_uuid(),
    account_id uuid not null references public.accounts(id) on delete cascade,
    snapshot_date date not null,
    market_value numeric not null,
    cash_balance numeric not null,
    total_value numeric not null,
    currency text not null default 'USD',
    source text not null default 'manual',
    created_at timestamptz not null default now(),
    constraint portfolio_snapshots_nonnegative_market_value check (market_value >= 0),
    constraint portfolio_snapshots_nonnegative_total_value check (total_value >= 0),
    constraint portfolio_snapshots_total_value_matches check (
        total_value = market_value + cash_balance
    ),
    constraint portfolio_snapshots_currency_upper check (currency = upper(currency)),
    constraint portfolio_snapshots_unique_account_date unique (account_id, snapshot_date)
);

create table if not exists public.holding_snapshots (
    id uuid primary key default gen_random_uuid(),
    snapshot_id uuid not null references public.portfolio_snapshots(id) on delete cascade,
    ticker text not null,
    quantity numeric,
    market_value numeric not null,
    price numeric,
    currency text not null default 'USD',
    constraint holding_snapshots_ticker_not_blank check (length(btrim(ticker)) > 0),
    constraint holding_snapshots_nonnegative_market_value check (market_value >= 0),
    constraint holding_snapshots_nonnegative_quantity check (quantity is null or quantity >= 0),
    constraint holding_snapshots_nonnegative_price check (price is null or price >= 0),
    constraint holding_snapshots_currency_upper check (currency = upper(currency)),
    constraint holding_snapshots_unique_snapshot_ticker unique (snapshot_id, ticker)
);

create table if not exists public.cashflows (
    id uuid primary key default gen_random_uuid(),
    account_id uuid not null references public.accounts(id) on delete cascade,
    cashflow_date date not null,
    settled_date date,
    amount numeric not null,
    currency text not null default 'USD',
    cashflow_type text not null,
    source text not null default 'manual',
    external_ref text,
    notes text,
    included_in_snapshot_id uuid references public.portfolio_snapshots(id) on delete set null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint cashflows_amount_nonzero check (amount <> 0),
    constraint cashflows_currency_upper check (currency = upper(currency)),
    constraint cashflows_supported_type check (
        cashflow_type in (
            'deposit',
            'withdrawal',
            'dividend',
            'interest',
            'fee',
            'tax',
            'transfer'
        )
    ),
    constraint cashflows_settlement_after_cashflow check (
        settled_date is null or settled_date >= cashflow_date
    ),
    constraint cashflows_external_ref_unique unique (account_id, source, external_ref)
);

create table if not exists public.recommendation_runs (
    id uuid primary key default gen_random_uuid(),
    account_id uuid not null references public.accounts(id) on delete cascade,
    run_id text not null unique,
    as_of_date date not null,
    data_as_of_date date not null,
    model_version text not null,
    ml_score_scale numeric not null,
    config_hash text not null,
    status text not null default 'completed',
    created_at timestamptz not null default now(),
    constraint recommendation_runs_run_id_not_blank check (length(btrim(run_id)) > 0),
    constraint recommendation_runs_positive_score_scale check (ml_score_scale > 0)
);

create table if not exists public.recommendation_lines (
    id uuid primary key default gen_random_uuid(),
    recommendation_run_id uuid not null references public.recommendation_runs(id) on delete cascade,
    ticker text not null,
    security text,
    gics_sector text,
    current_weight numeric,
    target_weight numeric,
    trade_weight numeric,
    trade_notional numeric,
    commission_amount numeric,
    cash_required_weight numeric,
    cash_released_weight numeric,
    deposit_used_amount numeric,
    cash_after_trade_amount numeric,
    action text,
    reason_code text,
    expected_return numeric,
    volatility numeric,
    constraint recommendation_lines_ticker_not_blank check (length(btrim(ticker)) > 0),
    constraint recommendation_lines_unique_run_ticker unique (recommendation_run_id, ticker)
);

create table if not exists public.performance_snapshots (
    id uuid primary key default gen_random_uuid(),
    account_id uuid not null references public.accounts(id) on delete cascade,
    as_of_date date not null,
    account_total_value numeric not null,
    total_deposits numeric not null,
    net_external_cashflow numeric not null,
    account_time_weighted_return numeric,
    account_money_weighted_return numeric,
    spy_same_cashflow_value numeric,
    spy_time_weighted_return numeric,
    spy_money_weighted_return numeric,
    active_value numeric,
    active_return numeric,
    created_at timestamptz not null default now(),
    constraint performance_snapshots_nonnegative_account_value check (account_total_value >= 0),
    constraint performance_snapshots_nonnegative_total_deposits check (total_deposits >= 0),
    constraint performance_snapshots_unique_account_date unique (account_id, as_of_date)
);

create index if not exists cashflows_account_date_idx
    on public.cashflows (account_id, cashflow_date);

create index if not exists cashflows_account_settled_date_idx
    on public.cashflows (account_id, settled_date);

create index if not exists portfolio_snapshots_account_date_idx
    on public.portfolio_snapshots (account_id, snapshot_date desc);

create index if not exists recommendation_runs_account_date_idx
    on public.recommendation_runs (account_id, as_of_date desc);

create index if not exists performance_snapshots_account_date_idx
    on public.performance_snapshots (account_id, as_of_date desc);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists accounts_set_updated_at on public.accounts;
create trigger accounts_set_updated_at
before update on public.accounts
for each row execute function public.set_updated_at();

drop trigger if exists cashflows_set_updated_at on public.cashflows;
create trigger cashflows_set_updated_at
before update on public.cashflows
for each row execute function public.set_updated_at();

alter table public.accounts enable row level security;
alter table public.cashflows enable row level security;
alter table public.portfolio_snapshots enable row level security;
alter table public.holding_snapshots enable row level security;
alter table public.recommendation_runs enable row level security;
alter table public.recommendation_lines enable row level security;
alter table public.performance_snapshots enable row level security;
