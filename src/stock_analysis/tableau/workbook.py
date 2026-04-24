from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from lxml import etree as ET

TABLEAU_WORKBOOK_VERSION = "18.1"
TABLEAU_SOURCE_BUILD = "2025.3.2 (20253.26.0109.0333)"
INTERNAL_DATASOURCE_NAME = "federated.portfolio_dashboard_mart"
TABLEAU_USER_NS = "http://www.tableausoftware.com/xml/user"

FieldRole = Literal["dimension", "measure"]
FieldType = Literal["nominal", "quantitative"]
Derivation = Literal["None", "Sum", "Avg", "Max"]


@dataclass(frozen=True)
class TableauField:
    name: str
    datatype: str
    role: FieldRole
    field_type: FieldType
    caption: str | None = None
    formula: str | None = None


@dataclass(frozen=True)
class ShelfField:
    field: str
    derivation: Derivation = "None"


@dataclass(frozen=True)
class WorksheetSpec:
    name: str
    mark_class: str
    rows: tuple[ShelfField, ...] = ()
    cols: tuple[ShelfField, ...] = ()
    encodings: tuple[tuple[str, ShelfField], ...] = ()


@dataclass(frozen=True)
class PortfolioWorkbookSpec:
    server_url: str
    datasource_name: str = "portfolio_dashboard_mart"
    table_name: str = "portfolio_dashboard_mart"
    workbook_name: str = "portfolio_recommendations"
    project_name: str = "Default"
    site_name: str | None = None
    width: int = 1280
    height: int = 850


FIELDS: dict[str, TableauField] = {
    "run_id": TableauField("run_id", "string", "dimension", "nominal", "Run ID"),
    "as_of_date": TableauField("as_of_date", "date", "dimension", "nominal", "As Of Date"),
    "ticker": TableauField("ticker", "string", "dimension", "nominal", "Ticker"),
    "security": TableauField("security", "string", "dimension", "nominal", "Security"),
    "gics_sector": TableauField("gics_sector", "string", "dimension", "nominal", "GICS Sector"),
    "forecast_score": TableauField(
        "forecast_score", "real", "measure", "quantitative", "Forecast Score"
    ),
    "volatility": TableauField("volatility", "real", "measure", "quantitative", "Volatility"),
    "target_weight": TableauField(
        "target_weight", "real", "measure", "quantitative", "Target Weight"
    ),
    "current_weight": TableauField(
        "current_weight", "real", "measure", "quantitative", "Current Weight"
    ),
    "trade_weight": TableauField("trade_weight", "real", "measure", "quantitative", "Trade Weight"),
    "trade_abs_weight": TableauField(
        "trade_abs_weight", "real", "measure", "quantitative", "Trade Abs Weight"
    ),
    "rebalance_required": TableauField(
        "rebalance_required", "boolean", "dimension", "nominal", "Rebalance Required"
    ),
    "estimated_commission_weight": TableauField(
        "estimated_commission_weight",
        "real",
        "measure",
        "quantitative",
        "Estimated Commission Weight",
    ),
    "net_trade_weight_after_commission": TableauField(
        "net_trade_weight_after_commission",
        "real",
        "measure",
        "quantitative",
        "Net Trade Weight After Commission",
    ),
    "cash_required_weight": TableauField(
        "cash_required_weight", "real", "measure", "quantitative", "Cash Required Weight"
    ),
    "cash_released_weight": TableauField(
        "cash_released_weight", "real", "measure", "quantitative", "Cash Released Weight"
    ),
    "current_weight_label": TableauField(
        "current_weight_label", "string", "dimension", "nominal", "Current Weight Label"
    ),
    "target_weight_label": TableauField(
        "target_weight_label", "string", "dimension", "nominal", "Target Weight Label"
    ),
    "trade_weight_label": TableauField(
        "trade_weight_label", "string", "dimension", "nominal", "Trade Weight Label"
    ),
    "estimated_commission_weight_label": TableauField(
        "estimated_commission_weight_label",
        "string",
        "dimension",
        "nominal",
        "Estimated Commission Weight Label",
    ),
    "selected": TableauField("selected", "boolean", "dimension", "nominal", "Selected"),
    "scatter_size": TableauField("scatter_size", "real", "measure", "quantitative", "Scatter Size"),
    "action": TableauField("action", "string", "dimension", "nominal", "Action"),
    "reason_code": TableauField("reason_code", "string", "dimension", "nominal", "Reason Code"),
    "sector_target_weight": TableauField(
        "sector_target_weight", "real", "measure", "quantitative", "Sector Target Weight"
    ),
    "portfolio_expected_return": TableauField(
        "portfolio_expected_return",
        "real",
        "measure",
        "quantitative",
        "Portfolio Forecast Score",
    ),
    "portfolio_expected_volatility": TableauField(
        "portfolio_expected_volatility",
        "real",
        "measure",
        "quantitative",
        "Portfolio Volatility",
    ),
    "portfolio_return_per_vol": TableauField(
        "portfolio_return_per_vol", "real", "measure", "quantitative", "Return per Vol"
    ),
    "portfolio_num_holdings": TableauField(
        "portfolio_num_holdings", "integer", "measure", "quantitative", "# Holdings"
    ),
    "portfolio_max_weight": TableauField(
        "portfolio_max_weight", "real", "measure", "quantitative", "Max Weight"
    ),
    "portfolio_concentration_hhi": TableauField(
        "portfolio_concentration_hhi", "real", "measure", "quantitative", "Concentration HHI"
    ),
    "run_requested_as_of_date": TableauField(
        "run_requested_as_of_date", "date", "dimension", "nominal", "Requested Data Date"
    ),
    "run_data_as_of_date": TableauField(
        "run_data_as_of_date", "date", "dimension", "nominal", "Market Data Date"
    ),
    "is_data_date_lagged": TableauField(
        "is_data_date_lagged", "boolean", "dimension", "nominal", "Data Date Lagged"
    ),
    "data_date_status": TableauField(
        "data_date_status", "string", "dimension", "nominal", "Data Date Status"
    ),
    "run_created_at_utc": TableauField(
        "run_created_at_utc", "string", "dimension", "nominal", "Run Created At UTC"
    ),
    "run_config_hash": TableauField(
        "run_config_hash", "string", "dimension", "nominal", "Run Config Hash"
    ),
    "run_config_hash_short": TableauField(
        "run_config_hash_short", "string", "dimension", "nominal", "Config Hash"
    ),
    "portfolio_footer_label": TableauField(
        "portfolio_footer_label",
        "string",
        "dimension",
        "nominal",
        "Portfolio Footer Label",
        (
            '"Run " + STR([run_id]) + " | requested " + STR([run_requested_as_of_date]) '
            '+ " | data " + STR([run_data_as_of_date]) + " | " + STR([data_date_status]) '
            '+ " | config " + STR([run_config_hash_short])'
        ),
    ),
    "holding_ticker": TableauField(
        "holding_ticker",
        "string",
        "dimension",
        "nominal",
        "Holding Ticker",
        "IF [target_weight] > 0 THEN [ticker] END",
    ),
    "holding_security": TableauField(
        "holding_security",
        "string",
        "dimension",
        "nominal",
        "Holding Security",
        "IF [target_weight] > 0 THEN [security] END",
    ),
    "holding_weight": TableauField(
        "holding_weight",
        "real",
        "measure",
        "quantitative",
        "Holding Weight",
        "IF [target_weight] > 0 THEN [target_weight] END",
    ),
    "holding_weight_label": TableauField(
        "holding_weight_label",
        "string",
        "dimension",
        "nominal",
        "Holding Weight Label",
        "IF [target_weight] > 0 THEN [target_weight_label] END",
    ),
    "trade_ticker": TableauField(
        "trade_ticker",
        "string",
        "dimension",
        "nominal",
        "Trade Ticker",
        "IF [rebalance_required] THEN [ticker] END",
    ),
    "trade_size": TableauField(
        "trade_size",
        "real",
        "measure",
        "quantitative",
        "Trade Size",
        "IF [rebalance_required] THEN [trade_abs_weight] END",
    ),
    "trade_description": TableauField(
        "trade_description",
        "string",
        "dimension",
        "nominal",
        "Trade Description",
        'IF [rebalance_required] THEN [action] + " " + [ticker] + " " + [trade_weight_label] END',
    ),
}


WORKSHEETS: tuple[WorksheetSpec, ...] = (
    WorksheetSpec(
        name="KPI Data As Of",
        mark_class="Text",
        encodings=(("text", ShelfField("run_data_as_of_date")),),
    ),
    WorksheetSpec(
        name="KPI Holdings",
        mark_class="Text",
        encodings=(("text", ShelfField("portfolio_num_holdings", "Max")),),
    ),
    WorksheetSpec(
        name="KPI Forecast Score",
        mark_class="Text",
        encodings=(("text", ShelfField("portfolio_expected_return", "Max")),),
    ),
    WorksheetSpec(
        name="KPI Volatility",
        mark_class="Text",
        encodings=(("text", ShelfField("portfolio_expected_volatility", "Max")),),
    ),
    WorksheetSpec(
        name="KPI Return Vol",
        mark_class="Text",
        encodings=(("text", ShelfField("portfolio_return_per_vol", "Max")),),
    ),
    WorksheetSpec(
        name="KPI Weight Sum",
        mark_class="Text",
        encodings=(("text", ShelfField("target_weight", "Sum")),),
    ),
    WorksheetSpec(
        name="Holdings by Weight",
        mark_class="Bar",
        rows=(ShelfField("holding_ticker"), ShelfField("holding_security")),
        cols=(ShelfField("holding_weight", "Sum"),),
        encodings=(
            ("color", ShelfField("gics_sector")),
            ("text", ShelfField("holding_weight_label")),
            ("tooltip", ShelfField("reason_code")),
        ),
    ),
    WorksheetSpec(
        name="Trade Tickets",
        mark_class="Bar",
        rows=(ShelfField("action"), ShelfField("trade_ticker"), ShelfField("security")),
        cols=(ShelfField("trade_size", "Sum"),),
        encodings=(
            ("color", ShelfField("action")),
            ("text", ShelfField("trade_description")),
            ("tooltip", ShelfField("estimated_commission_weight_label")),
        ),
    ),
    WorksheetSpec(
        name="Sector Allocation",
        mark_class="Square",
        encodings=(
            ("color", ShelfField("gics_sector")),
            ("size", ShelfField("target_weight", "Sum")),
            ("text", ShelfField("gics_sector")),
            ("tooltip", ShelfField("sector_target_weight", "Max")),
        ),
    ),
    WorksheetSpec(
        name="Risk Forecast Scatter",
        mark_class="Circle",
        rows=(ShelfField("forecast_score", "Avg"),),
        cols=(ShelfField("volatility", "Avg"),),
        encodings=(
            ("color", ShelfField("gics_sector")),
            ("size", ShelfField("scatter_size", "Sum")),
            ("lod", ShelfField("ticker")),
            ("text", ShelfField("ticker")),
            ("tooltip", ShelfField("security")),
        ),
    ),
    WorksheetSpec(
        name="Freshness Footer",
        mark_class="Text",
        encodings=(("text", ShelfField("portfolio_footer_label")),),
    ),
)


def write_portfolio_workbook(spec: PortfolioWorkbookSpec, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(build_portfolio_workbook_xml(spec))
    tree.write(
        str(output_path),
        encoding="UTF-8",
        xml_declaration=True,
        pretty_print=True,
    )
    return output_path


def build_portfolio_workbook_xml(spec: PortfolioWorkbookSpec) -> ET._Element:
    root = ET.Element(
        "workbook",
        nsmap={"user": TABLEAU_USER_NS},
        attrib={
            "original-version": TABLEAU_WORKBOOK_VERSION,
            "source-build": TABLEAU_SOURCE_BUILD,
            "source-platform": "mac",
            "version": TABLEAU_WORKBOOK_VERSION,
        },
    )
    root.append(
        ET.Comment("Generated by stock-analysis. Open in Tableau Desktop to validate and refine.")
    )
    _add_manifest(root)
    _add_preferences(root)
    _add_datasources(root, spec)
    _add_worksheets(root)
    _add_dashboards(root, spec)
    _add_windows(root)
    return root


def _add_manifest(root: ET._Element) -> None:
    manifest = ET.SubElement(root, "document-format-change-manifest")
    for tag in (
        "AnimationOnByDefault",
        "MarkAnimation",
        "ObjectModelEncapsulateLegacy",
        "ObjectModelTableType",
        "SchemaViewerObjectModel",
        "SheetIdentifierTracking",
        "WindowsPersistSimpleIdentifiers",
    ):
        ET.SubElement(manifest, tag)


def _add_preferences(root: ET._Element) -> None:
    preferences = ET.SubElement(root, "preferences")
    ET.SubElement(preferences, "preference", name="ui.encoding.shelf.height", value="24")
    ET.SubElement(preferences, "preference", name="ui.shelf.height", value="26")


def _add_datasources(root: ET._Element, spec: PortfolioWorkbookSpec) -> None:
    datasources = ET.SubElement(root, "datasources")
    datasource = ET.SubElement(
        datasources,
        "datasource",
        caption=spec.datasource_name,
        inline="true",
        name=INTERNAL_DATASOURCE_NAME,
        version=TABLEAU_WORKBOOK_VERSION,
    )
    ET.SubElement(
        datasource,
        "repository-location",
        {
            "derived-from": f"/dataserver/{spec.datasource_name}?rev=1.0",
            "id": spec.datasource_name,
            "path": "/datasources",
            "revision": "1.0",
        },
    )
    connection = ET.SubElement(
        datasource,
        "connection",
        channel="https",
        class_="sqlproxy",
        dbname=spec.datasource_name,
        directory="/dataserver",
        port="82",
        server=spec.server_url,
        username="",
    )
    # lxml cannot create the reserved Python keyword attribute via keyword args.
    connection.attrib["class"] = connection.attrib.pop("class_")
    ET.SubElement(
        connection,
        "relation",
        name=spec.table_name,
        table=f"[{spec.table_name}]",
        type="table",
    )
    for field in FIELDS.values():
        _add_field_column(datasource, field)
    ET.SubElement(
        datasource,
        "layout",
        {"dim-ordering": "alphabetic", "measure-ordering": "alphabetic", "show-structure": "true"},
    )
    ET.SubElement(datasource, "date-options", {"start-of-week": "monday"})
    object_graph = ET.SubElement(datasource, "object-graph")
    objects = ET.SubElement(object_graph, "objects")
    obj = ET.SubElement(
        objects,
        "object",
        caption=spec.table_name,
        id="portfolio_dashboard_mart_object",
    )
    properties = ET.SubElement(obj, "properties", context="")
    ET.SubElement(
        properties,
        "relation",
        name=spec.table_name,
        table=f"[{spec.table_name}]",
        type="table",
    )


def _add_worksheets(root: ET._Element) -> None:
    worksheets = ET.SubElement(root, "worksheets")
    for worksheet in WORKSHEETS:
        _add_worksheet(worksheets, worksheet)


def _add_worksheet(parent: ET._Element, worksheet_spec: WorksheetSpec) -> None:
    worksheet = ET.SubElement(parent, "worksheet", name=worksheet_spec.name)
    table = ET.SubElement(worksheet, "table")
    view = ET.SubElement(table, "view")
    datasources = ET.SubElement(view, "datasources")
    ET.SubElement(
        datasources,
        "datasource",
        caption="portfolio_dashboard_mart",
        name=INTERNAL_DATASOURCE_NAME,
    )
    dependencies = ET.SubElement(
        view,
        "datasource-dependencies",
        datasource=INTERNAL_DATASOURCE_NAME,
    )
    for field_name in sorted(_worksheet_field_names(worksheet_spec)):
        _add_field_column(dependencies, FIELDS[field_name])
    for shelf_field in _worksheet_shelf_fields(worksheet_spec):
        ET.SubElement(
            dependencies,
            "column-instance",
            column=_field_name(shelf_field.field),
            derivation=shelf_field.derivation,
            name=_instance_name(shelf_field),
            pivot="key",
            type=_instance_type(shelf_field),
        )
    ET.SubElement(view, "aggregation", value="true")
    ET.SubElement(table, "style")
    panes = ET.SubElement(table, "panes")
    pane = ET.SubElement(
        panes, "pane", {"selection-relaxation-option": "selection-relaxation-disallow"}
    )
    pane_view = ET.SubElement(pane, "view")
    ET.SubElement(pane_view, "breakdown", value="auto")
    ET.SubElement(pane, "mark", {"class": worksheet_spec.mark_class})
    encodings = ET.SubElement(pane, "encodings")
    for encoding_name, shelf_field in worksheet_spec.encodings:
        ET.SubElement(encodings, encoding_name, column=_qualified_instance(shelf_field))
    _add_mark_label_style(pane)
    rows = ET.SubElement(table, "rows")
    rows.text = _shelf_text(worksheet_spec.rows)
    cols = ET.SubElement(table, "cols")
    cols.text = _shelf_text(worksheet_spec.cols)
    ET.SubElement(worksheet, "simple-id", uuid=_stable_uuid(f"worksheet:{worksheet_spec.name}"))


def _add_mark_label_style(pane: ET._Element) -> None:
    style = ET.SubElement(pane, "style")
    style_rule = ET.SubElement(style, "style-rule", element="mark")
    ET.SubElement(style_rule, "format", attr="mark-labels-show", value="true")
    ET.SubElement(style_rule, "format", attr="mark-labels-cull", value="true")


def _add_dashboards(root: ET._Element, spec: PortfolioWorkbookSpec) -> None:
    dashboards = ET.SubElement(root, "dashboards")
    dashboard = ET.SubElement(dashboards, "dashboard", name="Portfolio Recommendations")
    ET.SubElement(dashboard, "style")
    ET.SubElement(
        dashboard,
        "size",
        maxheight=str(spec.height),
        maxwidth=str(spec.width),
        minheight=str(spec.height),
        minwidth=str(spec.width),
        **{"sizing-mode": "fixed"},
    )
    zones = ET.SubElement(dashboard, "zones")
    root_zone = _layout_zone(zones, 1, 0, 0, 100000, 100000, "vert")
    kpi_zone = _layout_zone(root_zone, 2, 0, 0, 100000, 14000, "horz")
    kpi_names = [
        "KPI Data As Of",
        "KPI Holdings",
        "KPI Forecast Score",
        "KPI Volatility",
        "KPI Return Vol",
        "KPI Weight Sum",
    ]
    kpi_width = 100000 // len(kpi_names)
    for index, sheet_name in enumerate(kpi_names):
        _sheet_zone(kpi_zone, 3 + index, kpi_width * index, 0, kpi_width, 14000, sheet_name)

    body_zone = _layout_zone(root_zone, 20, 0, 14000, 100000, 76000, "vert")
    main_zone = _layout_zone(body_zone, 21, 0, 14000, 100000, 52000, "horz")
    _sheet_zone(main_zone, 22, 0, 14000, 43000, 52000, "Holdings by Weight")
    right_zone = _layout_zone(main_zone, 23, 43000, 14000, 57000, 52000, "vert")
    _sheet_zone(right_zone, 24, 43000, 14000, 57000, 25000, "Sector Allocation")
    _sheet_zone(right_zone, 25, 43000, 39000, 57000, 27000, "Risk Forecast Scatter")
    _sheet_zone(body_zone, 26, 0, 66000, 100000, 24000, "Trade Tickets")

    _sheet_zone(root_zone, 30, 0, 90000, 100000, 10000, "Freshness Footer")
    _append_layout_zone_styles(root_zone)
    ET.SubElement(dashboard, "simple-id", uuid=_stable_uuid("dashboard:Portfolio Recommendations"))


def _add_windows(root: ET._Element) -> None:
    windows = ET.SubElement(
        root, "windows", {"saved-dpi-scale-factor": "1.25", "source-height": "37"}
    )
    for worksheet in WORKSHEETS:
        window = ET.SubElement(windows, "window", {"class": "worksheet", "name": worksheet.name})
        cards = ET.SubElement(window, "cards")
        left = ET.SubElement(cards, "edge", name="left")
        strip = ET.SubElement(left, "strip", size="160")
        for card_type in ("pages", "filters", "marks"):
            ET.SubElement(strip, "card", type=card_type)
        top = ET.SubElement(cards, "edge", name="top")
        for card_type in ("columns", "rows", "title"):
            top_strip = ET.SubElement(top, "strip", size="2147483647")
            ET.SubElement(top_strip, "card", type=card_type)
        ET.SubElement(cards, "edge", name="right")
        ET.SubElement(cards, "edge", name="bottom")
        ET.SubElement(window, "simple-id", uuid=_stable_uuid(f"window:{worksheet.name}"))

    dashboard_window = ET.SubElement(
        windows,
        "window",
        {"class": "dashboard", "name": "Portfolio Recommendations"},
    )
    viewpoints = ET.SubElement(dashboard_window, "viewpoints")
    for worksheet in WORKSHEETS:
        ET.SubElement(viewpoints, "viewpoint", name=worksheet.name)
    ET.SubElement(dashboard_window, "active", id="-1")
    ET.SubElement(
        dashboard_window, "simple-id", uuid=_stable_uuid("window:Portfolio Recommendations")
    )


def _layout_zone(
    parent: ET._Element,
    zone_id: int,
    x: int,
    y: int,
    width: int,
    height: int,
    direction: Literal["horz", "vert"],
) -> ET._Element:
    zone = ET.SubElement(
        parent,
        "zone",
        id=str(zone_id),
        x=str(x),
        y=str(y),
        w=str(width),
        h=str(height),
        **{
            "type-v2": "layout-flow",
            "param": direction,
            "layout-strategy-id": "distribute-evenly",
        },
    )
    return zone


def _sheet_zone(
    parent: ET._Element,
    zone_id: int,
    x: int,
    y: int,
    width: int,
    height: int,
    sheet_name: str,
) -> ET._Element:
    zone = ET.SubElement(
        parent,
        "zone",
        id=str(zone_id),
        x=str(x),
        y=str(y),
        w=str(width),
        h=str(height),
        name=sheet_name,
        **{"show-title": "true"},
    )
    _add_zone_style(zone)
    return zone


def _add_zone_style(zone: ET._Element) -> None:
    zone_style = ET.SubElement(zone, "zone-style")
    ET.SubElement(zone_style, "format", attr="border-color", value="#000000")
    ET.SubElement(zone_style, "format", attr="border-style", value="none")
    ET.SubElement(zone_style, "format", attr="border-width", value="0")


def _append_layout_zone_styles(zone: ET._Element) -> None:
    for child_zone in zone.findall("./zone"):
        _append_layout_zone_styles(child_zone)
    if zone.attrib.get("type-v2") == "layout-flow" and zone.find("./zone-style") is None:
        _add_zone_style(zone)


def _add_field_column(parent: ET._Element, field: TableauField) -> None:
    attrs = {
        "datatype": field.datatype,
        "name": _field_name(field.name),
        "role": field.role,
        "type": field.field_type,
    }
    if field.caption:
        attrs["caption"] = field.caption
    column = ET.SubElement(parent, "column", attrs)
    if field.formula:
        ET.SubElement(column, "calculation", {"class": "tableau", "formula": field.formula})


def _worksheet_field_names(worksheet: WorksheetSpec) -> set[str]:
    return {shelf_field.field for shelf_field in _worksheet_shelf_fields(worksheet)}


def _worksheet_shelf_fields(worksheet: WorksheetSpec) -> tuple[ShelfField, ...]:
    fields = [*worksheet.rows, *worksheet.cols, *(shelf for _, shelf in worksheet.encodings)]
    deduped: dict[tuple[str, Derivation], ShelfField] = {}
    for field in fields:
        deduped[(field.field, field.derivation)] = field
    return tuple(deduped.values())


def _field_name(field: str) -> str:
    return f"[{field}]"


def _instance_name(shelf_field: ShelfField) -> str:
    prefix = {
        "None": "none",
        "Sum": "sum",
        "Avg": "avg",
        "Max": "max",
    }[shelf_field.derivation]
    suffix = "nk" if shelf_field.derivation == "None" else "qk"
    return f"[{prefix}:{shelf_field.field}:{suffix}]"


def _qualified_instance(shelf_field: ShelfField) -> str:
    return f"[{INTERNAL_DATASOURCE_NAME}].{_instance_name(shelf_field)}"


def _instance_type(shelf_field: ShelfField) -> str:
    return "nominal" if shelf_field.derivation == "None" else "quantitative"


def _shelf_text(shelf_fields: tuple[ShelfField, ...]) -> str | None:
    if not shelf_fields:
        return None
    return " / ".join(_qualified_instance(field) for field in shelf_fields)


def _stable_uuid(key: str) -> str:
    return f"{{{str(uuid5(NAMESPACE_URL, f'stock-analysis-tableau:{key}')).upper()}}}"
