alter table public.accounts
    add column if not exists owner_id uuid references auth.users(id) on delete set null;

create index if not exists accounts_owner_id_idx
    on public.accounts (owner_id);

alter table public.portfolio_snapshots
    drop constraint if exists portfolio_snapshots_total_value_matches;

alter table public.portfolio_snapshots
    add constraint portfolio_snapshots_total_value_matches check (
        abs(total_value - (market_value + cash_balance))
            <= greatest(0.01, abs(market_value + cash_balance) * 0.000001)
    );

do $$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'accounts'
          and policyname = 'accounts_owner_access'
    ) then
        create policy accounts_owner_access
            on public.accounts
            for all
            to authenticated
            using (owner_id = auth.uid())
            with check (owner_id = auth.uid());
    end if;
end;
$$;

do $$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'cashflows'
          and policyname = 'cashflows_owner_access'
    ) then
        create policy cashflows_owner_access
            on public.cashflows
            for all
            to authenticated
            using (
                exists (
                    select 1
                    from public.accounts
                    where accounts.id = cashflows.account_id
                      and accounts.owner_id = auth.uid()
                )
            )
            with check (
                exists (
                    select 1
                    from public.accounts
                    where accounts.id = cashflows.account_id
                      and accounts.owner_id = auth.uid()
                )
            );
    end if;
end;
$$;

do $$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'portfolio_snapshots'
          and policyname = 'portfolio_snapshots_owner_access'
    ) then
        create policy portfolio_snapshots_owner_access
            on public.portfolio_snapshots
            for all
            to authenticated
            using (
                exists (
                    select 1
                    from public.accounts
                    where accounts.id = portfolio_snapshots.account_id
                      and accounts.owner_id = auth.uid()
                )
            )
            with check (
                exists (
                    select 1
                    from public.accounts
                    where accounts.id = portfolio_snapshots.account_id
                      and accounts.owner_id = auth.uid()
                )
            );
    end if;
end;
$$;

do $$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'holding_snapshots'
          and policyname = 'holding_snapshots_owner_access'
    ) then
        create policy holding_snapshots_owner_access
            on public.holding_snapshots
            for all
            to authenticated
            using (
                exists (
                    select 1
                    from public.portfolio_snapshots
                    join public.accounts
                      on accounts.id = portfolio_snapshots.account_id
                    where portfolio_snapshots.id = holding_snapshots.snapshot_id
                      and accounts.owner_id = auth.uid()
                )
            )
            with check (
                exists (
                    select 1
                    from public.portfolio_snapshots
                    join public.accounts
                      on accounts.id = portfolio_snapshots.account_id
                    where portfolio_snapshots.id = holding_snapshots.snapshot_id
                      and accounts.owner_id = auth.uid()
                )
            );
    end if;
end;
$$;

do $$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'recommendation_runs'
          and policyname = 'recommendation_runs_owner_access'
    ) then
        create policy recommendation_runs_owner_access
            on public.recommendation_runs
            for all
            to authenticated
            using (
                exists (
                    select 1
                    from public.accounts
                    where accounts.id = recommendation_runs.account_id
                      and accounts.owner_id = auth.uid()
                )
            )
            with check (
                exists (
                    select 1
                    from public.accounts
                    where accounts.id = recommendation_runs.account_id
                      and accounts.owner_id = auth.uid()
                )
            );
    end if;
end;
$$;

do $$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'recommendation_lines'
          and policyname = 'recommendation_lines_owner_access'
    ) then
        create policy recommendation_lines_owner_access
            on public.recommendation_lines
            for all
            to authenticated
            using (
                exists (
                    select 1
                    from public.recommendation_runs
                    join public.accounts
                      on accounts.id = recommendation_runs.account_id
                    where recommendation_runs.id = recommendation_lines.recommendation_run_id
                      and accounts.owner_id = auth.uid()
                )
            )
            with check (
                exists (
                    select 1
                    from public.recommendation_runs
                    join public.accounts
                      on accounts.id = recommendation_runs.account_id
                    where recommendation_runs.id = recommendation_lines.recommendation_run_id
                      and accounts.owner_id = auth.uid()
                )
            );
    end if;
end;
$$;

do $$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'performance_snapshots'
          and policyname = 'performance_snapshots_owner_access'
    ) then
        create policy performance_snapshots_owner_access
            on public.performance_snapshots
            for all
            to authenticated
            using (
                exists (
                    select 1
                    from public.accounts
                    where accounts.id = performance_snapshots.account_id
                      and accounts.owner_id = auth.uid()
                )
            )
            with check (
                exists (
                    select 1
                    from public.accounts
                    where accounts.id = performance_snapshots.account_id
                      and accounts.owner_id = auth.uid()
                )
            );
    end if;
end;
$$;
