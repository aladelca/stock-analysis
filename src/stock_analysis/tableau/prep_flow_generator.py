from __future__ import annotations

from pathlib import Path


def generate_prep_flow_if_available(output_path: Path) -> Path | None:
    try:
        import cwprep  # noqa: F401
    except ImportError:
        return None

    # cwprep support is intentionally optional. The checked-in flow spec remains the
    # authoritative artifact until a generated .tfl is validated in Tableau Prep Builder.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Generated Tableau Prep flow placeholder\n"
        "# Validate cwprep output manually before replacing this placeholder with a .tfl.\n",
        encoding="utf-8",
    )
    return output_path
