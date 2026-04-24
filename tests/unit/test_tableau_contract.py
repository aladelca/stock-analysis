from __future__ import annotations

from stock_analysis.tableau.prep_contract import PREP_INPUTS


def test_prep_inputs_define_required_columns() -> None:
    names = {item.name for item in PREP_INPUTS}

    assert "portfolio_recommendations" in names
    assert all(item.required_columns for item in PREP_INPUTS)
