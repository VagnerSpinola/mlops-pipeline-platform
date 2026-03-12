"""Guardrails for deciding whether a model is deployment-ready."""

from __future__ import annotations


def validate_model_performance(
    metrics: dict[str, float],
    minimum_roc_auc: float,
    minimum_f1_score: float,
) -> list[str]:
    """Return validation issues when a model misses performance gates."""
    issues: list[str] = []
    if metrics["roc_auc"] < minimum_roc_auc:
        issues.append(
            f"ROC-AUC below threshold: {metrics['roc_auc']:.3f} < {minimum_roc_auc:.3f}"
        )
    if metrics["f1_score"] < minimum_f1_score:
        issues.append(
            f"F1-score below threshold: {metrics['f1_score']:.3f} < {minimum_f1_score:.3f}"
        )
    return issues