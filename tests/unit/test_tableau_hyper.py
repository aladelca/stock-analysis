from __future__ import annotations

import pandas as pd
import pytest

from stock_analysis.tableau.hyper import export_hyper_if_available


def test_hyper_export_handles_object_backed_boolean_columns(tmp_path) -> None:
    pytest.importorskip("tableauhyperapi")
    frame = pd.DataFrame(
        {
            "ticker": ["SPY", "AAPL"],
            "no_trade_band_applied": pd.Series([False, True], dtype="object"),
        }
    )

    exported = export_hyper_if_available(frame, tmp_path / "nullable_bool.hyper")

    assert exported == tmp_path / "nullable_bool.hyper"
    assert exported.exists()
