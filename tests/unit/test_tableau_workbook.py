from __future__ import annotations

from pathlib import Path

from lxml import etree as ET

from stock_analysis.tableau.workbook import (
    INTERNAL_DATASOURCE_NAME,
    PortfolioWorkbookSpec,
    write_portfolio_workbook,
)


def test_write_portfolio_workbook_generates_twb_xml(tmp_path: Path) -> None:
    output = tmp_path / "portfolio_recommendations.twb"
    spec = PortfolioWorkbookSpec(
        server_url="https://us-east-1.online.tableau.com",
        datasource_name="portfolio_dashboard_mart",
        site_name="aladelca",
    )

    written = write_portfolio_workbook(spec, output)

    assert written == output
    tree = ET.parse(str(output))
    root = tree.getroot()
    assert root.tag == "workbook"
    assert root.attrib["version"] == "18.1"

    connection = root.find(".//connection")
    assert connection is not None
    assert connection.attrib["class"] == "sqlproxy"
    assert connection.attrib["server"] == "https://us-east-1.online.tableau.com"
    assert connection.attrib["dbname"] == "portfolio_dashboard_mart"


def test_portfolio_workbook_contains_dashboard_sheets(tmp_path: Path) -> None:
    output = tmp_path / "portfolio_recommendations.twb"
    write_portfolio_workbook(
        PortfolioWorkbookSpec(server_url="https://tableau.example.com"),
        output,
    )
    root = ET.parse(str(output)).getroot()

    worksheet_names = {node.attrib["name"] for node in root.findall(".//worksheets/worksheet")}
    assert {
        "KPI Data As Of",
        "KPI Holdings",
        "KPI Forecast Score",
        "KPI Volatility",
        "KPI Return Vol",
        "KPI Weight Sum",
        "Holdings by Weight",
        "Sector Allocation",
        "Risk Forecast Scatter",
        "Freshness Footer",
    }.issubset(worksheet_names)

    dashboard = root.find(".//dashboards/dashboard[@name='Portfolio Recommendations']")
    assert dashboard is not None
    zone_names = {node.attrib["name"] for node in dashboard.findall(".//zone[@name]")}
    assert "Holdings by Weight" in zone_names
    assert "Sector Allocation" in zone_names
    assert "Risk Forecast Scatter" in zone_names


def test_portfolio_workbook_kpi_cards_do_not_use_axis_shelves(tmp_path: Path) -> None:
    output = tmp_path / "portfolio_recommendations.twb"
    write_portfolio_workbook(
        PortfolioWorkbookSpec(server_url="https://tableau.example.com"),
        output,
    )
    root = ET.parse(str(output)).getroot()

    kpi = root.find(".//worksheets/worksheet[@name='KPI Data As Of']/table")
    assert kpi is not None
    assert (kpi.findtext("./rows") or "") == ""
    assert (kpi.findtext("./cols") or "") == ""

    data_as_of_instance = kpi.find(".//column-instance[@column='[run_data_as_of_date]']")
    assert data_as_of_instance is not None
    assert data_as_of_instance.attrib["derivation"] == "None"
    assert data_as_of_instance.attrib["type"] == "nominal"

    footer = root.find(".//worksheets/worksheet[@name='Freshness Footer']/table")
    assert footer is not None
    assert (footer.findtext("./rows") or "") == ""
    footer_instance = footer.find(".//column-instance[@column='[portfolio_footer_label]']")
    assert footer_instance is not None
    assert footer_instance.attrib["derivation"] == "None"


def test_portfolio_workbook_writes_zone_style_after_child_zones(tmp_path: Path) -> None:
    output = tmp_path / "portfolio_recommendations.twb"
    write_portfolio_workbook(
        PortfolioWorkbookSpec(server_url="https://tableau.example.com"),
        output,
    )
    root = ET.parse(str(output)).getroot()

    for zone in root.findall(".//dashboards/dashboard//zone"):
        child_tags = [child.tag for child in zone]
        if "zone-style" not in child_tags:
            continue
        zone_style_index = child_tags.index("zone-style")
        later_tags = child_tags[zone_style_index + 1 :]
        assert "zone" not in later_tags


def test_portfolio_workbook_uses_expected_fields_and_calculation(tmp_path: Path) -> None:
    output = tmp_path / "portfolio_recommendations.twb"
    write_portfolio_workbook(
        PortfolioWorkbookSpec(server_url="https://tableau.example.com"),
        output,
    )
    root = ET.parse(str(output)).getroot()

    datasource = root.find(f".//datasource[@name='{INTERNAL_DATASOURCE_NAME}']")
    assert datasource is not None
    field_names = {node.attrib["name"] for node in datasource.findall("./column")}
    assert "[target_weight]" in field_names
    assert "[forecast_score]" in field_names
    assert "[portfolio_footer_label]" in field_names
    assert "[holding_ticker]" in field_names
    assert "[holding_weight]" in field_names

    footer_calc = datasource.find("./column[@name='[portfolio_footer_label]']/calculation")
    assert footer_calc is not None
    assert "[run_id]" in footer_calc.attrib["formula"]
    assert "[run_data_as_of_date]" in footer_calc.attrib["formula"]

    holding_calc = datasource.find("./column[@name='[holding_weight]']/calculation")
    assert holding_calc is not None
    assert "[target_weight] > 0" in holding_calc.attrib["formula"]
