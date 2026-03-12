"""Retraining trigger design placeholder with explicit decision reasons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mlops.drift.drift_monitor import DriftReport


@dataclass(slots=True)
class RetrainingDecision:
    """Decision object that explains why retraining was or was not triggered."""

    should_retrain: bool
    trigger_type: str
    reasons: list[str]
    reference_dataset_version: str
    current_dataset_version: str
    follow_up_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the retraining decision."""
        return asdict(self)


def evaluate_retraining_need(
    drift_report: DriftReport,
    reference_dataset_version: str,
    current_dataset_version: str,
    quality_summary: dict[str, Any],
    force_retrain: bool = False,
) -> RetrainingDecision:
    """Placeholder policy combining drift, freshness, and quality checks.

    In production, this decision would also include online performance regression,
    feature freshness SLAs, label backfill arrival, and business KPI deterioration.
    """
    reasons: list[str] = []
    follow_up_actions: list[str] = []
    trigger_type = "no_trigger"

    if force_retrain:
        reasons.append("Manual override requested retraining.")
        trigger_type = "manual_override"

    if drift_report.drift_detected:
        reasons.append("Feature drift threshold exceeded.")
        trigger_type = "data_drift"

    if reference_dataset_version != current_dataset_version:
        follow_up_actions.append("Review whether the incoming dataset reflects expected source-system changes.")

    if quality_summary.get("duplicate_rows", 0) > 0:
        follow_up_actions.append("Investigate duplicate records before promoting a retrained model.")

    if not reasons:
        reasons.append("No retraining trigger activated.")

    if drift_report.drift_detected:
        follow_up_actions.append("Compare offline challenger metrics to the current champion before promotion.")
        follow_up_actions.append("Run shadow or canary deployment before champion swap.")

    return RetrainingDecision(
        should_retrain=force_retrain or drift_report.drift_detected,
        trigger_type=trigger_type,
        reasons=reasons,
        reference_dataset_version=reference_dataset_version,
        current_dataset_version=current_dataset_version,
        follow_up_actions=follow_up_actions,
    )