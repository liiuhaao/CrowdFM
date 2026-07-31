from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from cfm.data.crowd_data import CrowdData

from .pipeline import masked_data
from .predictive_bootstrap import conditional_predictive_test
from .predictive_split import AnnotationAuditSplit, make_annotation_audit_split


@dataclass(frozen=True)
class PredictiveAuditConfig:
    audit_task_fraction: float = 1.0
    context_fraction: float = 0.5
    min_context_workers: int = 1
    min_audit_workers: int = 2
    min_worker_context_edges: int = 1
    min_pair_count: int = 1
    probability_clip: float = 1e-4
    variance_ridge: float = 1e-8
    num_bootstrap: int = 199
    alpha: float = 0.05
    require_trained_response_head: bool = True


def _response_head_is_untrained(model: torch.nn.Module) -> bool:
    trained = getattr(model, "response_head_trained", None)
    return trained is not None and not bool(trained)


def run_predictive_audit(
    model: torch.nn.Module,
    data: CrowdData,
    *,
    config: PredictiveAuditConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Audit q_k and edge-conditioned annotation emissions on held-out labels."""

    cfg = config or PredictiveAuditConfig()
    if not 0.0 < cfg.alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if cfg.num_bootstrap < 1:
        raise ValueError("num_bootstrap must be positive")
    minimum_joint_p_value = 2.0 / (cfg.num_bootstrap + 1.0)
    if minimum_joint_p_value > cfg.alpha:
        raise ValueError(
            "num_bootstrap is too small for the Bonferroni joint test at the "
            f"requested alpha: minimum attainable p-value is "
            f"{minimum_joint_p_value:.6f}"
        )
    if cfg.require_trained_response_head and _response_head_is_untrained(model):
        raise RuntimeError(
            "the conditional annotation head is marked untrained; train it "
            "before using its probabilities for a deployment audit"
        )

    split: AnnotationAuditSplit = make_annotation_audit_split(
        data.triple,
        data.num_task,
        num_worker=data.num_worker,
        audit_task_fraction=cfg.audit_task_fraction,
        context_fraction=cfg.context_fraction,
        min_context_workers=cfg.min_context_workers,
        min_audit_workers=cfg.min_audit_workers,
        min_worker_context_edges=cfg.min_worker_context_edges,
        seed=seed,
    )
    context_data = masked_data(data, split.context_edge_mask)
    audit_indices = split.audit_edge_indices
    query_workers = data.triple[0, audit_indices].long()
    observed_answers = data.triple[1, audit_indices].long()
    query_tasks = data.triple[2, audit_indices].long()

    model.eval()
    with torch.no_grad():
        output = model(
            context_data,
            query_workers=query_workers,
            query_tasks=query_tasks,
        )
    if "hat_task_option" not in output:
        raise KeyError("predictive audit requires output['hat_task_option']")
    if "hat_annotation_given_truth" not in output:
        raise KeyError(
            "predictive audit requires output['hat_annotation_given_truth']; use "
            "a trained PredictiveCFM or compatible model"
        )

    task_posterior = torch.softmax(output["hat_task_option"], dim=-1)
    emission_logits = output["hat_annotation_given_truth"]
    expected_shape = (audit_indices.numel(), data.num_option, data.num_option)
    if emission_logits.shape != expected_shape:
        raise ValueError(
            "hat_annotation_given_truth must have shape "
            f"{expected_shape}, got {tuple(emission_logits.shape)}"
        )
    emission_probabilities = torch.softmax(emission_logits, dim=-1)

    calibration = conditional_predictive_test(
        task_posterior,
        emission_probabilities,
        observed_answers,
        query_workers,
        query_tasks,
        num_worker=data.num_worker,
        num_bootstrap=cfg.num_bootstrap,
        seed=seed,
        min_pair_count=cfg.min_pair_count,
        probability_clip=cfg.probability_clip,
        variance_ridge=cfg.variance_ridge,
    )
    residual = calibration.observed
    supported_pairs = int(
        (residual.pair_counts >= cfg.min_pair_count).sum().item() // 2
    )

    return {
        "config": asdict(cfg),
        "seed": seed,
        "p_value": calibration.p_value,
        "minimum_joint_p_value": minimum_joint_p_value,
        "marginal_p_value": calibration.marginal_p_value,
        "dependence_p_value": calibration.dependence_p_value,
        "reject": calibration.p_value <= cfg.alpha,
        "marginal_statistic": residual.marginal_statistic,
        "dependence_statistic": residual.dependence_statistic,
        "num_supported_pairs": supported_pairs,
        "num_audit_edges": int(audit_indices.numel()),
        "num_context_edges": int(split.context_edge_mask.sum().item()),
        "task_posterior": residual.task_posterior,
        "emission_probabilities": residual.emission_probabilities,
        "marginal_response_probabilities": residual.marginal_probabilities,
        "residual": residual,
        "split": split,
        "bootstrap": calibration,
    }
