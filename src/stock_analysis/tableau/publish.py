from __future__ import annotations

import os
from pathlib import Path

from stock_analysis.config import TableauConfig


def publish_datasource_if_enabled(config: TableauConfig, datasource_path: Path) -> str | None:
    if not config.publish_enabled:
        return None
    if not datasource_path.exists():
        msg = f"Datasource path does not exist: {datasource_path}"
        raise FileNotFoundError(msg)
    try:
        import tableauserverclient as tsc
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = "Tableau publishing requires the optional tableau extra"
        raise RuntimeError(msg) from exc

    server_url = config.server_url or os.getenv("TABLEAU_SERVER_URL")
    site_name = config.site_name or os.getenv("TABLEAU_SITE_NAME", "")
    pat_name = os.getenv("TABLEAU_PAT_NAME")
    pat_value = os.getenv("TABLEAU_PAT_VALUE")
    if not server_url or not pat_name or not pat_value:
        msg = "Missing Tableau Server URL or PAT environment variables"
        raise ValueError(msg)

    auth = tsc.PersonalAccessTokenAuth(pat_name, pat_value, site_id=site_name)
    server = tsc.Server(server_url, use_server_version=True)
    with server.auth.sign_in(auth):
        project = next(
            (item for item in tsc.Pager(server.projects) if item.name == config.project_name),
            None,
        )
        if project is None:
            msg = f"Tableau project not found: {config.project_name}"
            raise ValueError(msg)
        datasource = tsc.DatasourceItem(project.id, name=config.datasource_name)
        published = server.datasources.publish(
            datasource,
            str(datasource_path),
            mode=tsc.Server.PublishMode.Overwrite,
        )
        return published.id


def publish_workbook_if_enabled(config: TableauConfig, workbook_path: Path) -> str | None:
    if not config.publish_enabled:
        return None
    if not workbook_path.exists():
        msg = f"Workbook path does not exist: {workbook_path}"
        raise FileNotFoundError(msg)
    try:
        import tableauserverclient as tsc
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = "Tableau workbook publishing requires the optional tableau extra"
        raise RuntimeError(msg) from exc

    server_url = config.server_url or os.getenv("TABLEAU_SERVER_URL")
    site_name = config.site_name or os.getenv("TABLEAU_SITE_NAME", "")
    pat_name = os.getenv("TABLEAU_PAT_NAME")
    pat_value = os.getenv("TABLEAU_PAT_VALUE")
    if not server_url or not pat_name or not pat_value:
        msg = "Missing Tableau Server URL or PAT environment variables"
        raise ValueError(msg)

    auth = tsc.PersonalAccessTokenAuth(pat_name, pat_value, site_id=site_name)
    server = tsc.Server(server_url, use_server_version=True)
    with server.auth.sign_in(auth):
        project = next(
            (item for item in tsc.Pager(server.projects) if item.name == config.project_name),
            None,
        )
        if project is None:
            msg = f"Tableau project not found: {config.project_name}"
            raise ValueError(msg)
        workbook = tsc.WorkbookItem(
            project.id,
            name=config.workbook_name,
            show_tabs=False,
        )
        published = server.workbooks.publish(
            workbook,
            str(workbook_path),
            mode=tsc.Server.PublishMode.Overwrite,
            skip_connection_check=False,
        )
        return published.id
