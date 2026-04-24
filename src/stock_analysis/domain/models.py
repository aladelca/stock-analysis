from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    as_of_date: date
    output_root: str
    recommendations_path: str
    risk_metrics_path: str
    sector_exposure_path: str
