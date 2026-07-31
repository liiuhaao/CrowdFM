from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from cfm.data.crowd_data import CrowdData

from .pipeline import masked_data
from .predictive_split import AnnotationAuditSplit, make_annotation_audit_split


@dataclass(frozen=True)
class PredictiveTrainingConfig:
    audit_task_fraction: float = 1.0
    context_fraction: float = 0.5
    min_context_workers: int = 1
    min_audit_workers: int = 2
    min_worker_context_edges: int = 0
    truth_loss_weight: float = 1.0
    annotation_loss_weight: float = 1.0


def predictive_training_loss(
    model: torch.nn.Module,
    data: CrowdData,
    *,
    config: PredictiveTrainingConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Joint task-truth and masked conditional-emission objective.

    For synthetic tasks with known truth, the held-out response supervises the
    emission row corresponding to that truth. For tasks without gold truth, the
    response loss marginalizes the emission rows using the model task posterior.
    """

    cfg = config or PredictiveTrainingConfig()
    if cfg.truth_loss_weight < 0 or cfg.annotation_loss_weight <= 0:
        raise ValueError(
            "truth loss weight must be nonnegative and annotation weight positive"
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

    output = model(
        context_data,
        query_workers=query_workers,
        query_tasks=query_tasks,
    )
    if "hat_annotation_given_truth" not in output:
        raise KeyError("model must output hat_annotation_given_truth for masked edges")
    emission_logits = output["hat_annotation_given_truth"]
    task_logits = output["hat_task_option"]
    device = emission_logits.device
    query_tasks = query_tasks.to(device)
    observed_answers = observed_answers.to(device)

    emission_log_probabilities = F.log_softmax(emission_logits, dim=-1)
    answer_log_probability_by_truth = emission_log_probabilities.gather(
        2,
        observed_answers[:, None, None].expand(-1, emission_logits.shape[1], 1),
    ).squeeze(2)
    task_log_posterior = F.log_softmax(task_logits[query_tasks], dim=-1)
    marginal_edge_losses = -torch.logsumexp(
        task_log_posterior + answer_log_probability_by_truth,
        dim=-1,
    )

    edge_losses = marginal_edge_losses
    task_y = getattr(data, "task_y", None)
    if isinstance(task_y, torch.Tensor):
        task_y = task_y.to(device)
        edge_truth = task_y[query_tasks]
        known = edge_truth != -1
        safe_truth = edge_truth.clamp_min(0)
        selected_rows = emission_logits[
            torch.arange(emission_logits.shape[0], device=device),
            safe_truth,
        ]
        supervised_edge_losses = F.cross_entropy(
            selected_rows,
            observed_answers,
            reduction="none",
        )
        edge_losses = torch.where(known, supervised_edge_losses, marginal_edge_losses)
    annotation_loss = edge_losses.mean()

    truth_loss = annotation_loss.new_zeros(())
    if cfg.truth_loss_weight > 0 and isinstance(task_y, torch.Tensor):
        valid = task_y != -1
        if torch.any(valid):
            truth_loss = F.cross_entropy(task_logits[valid], task_y[valid])

    total = (
        cfg.annotation_loss_weight * annotation_loss
        + cfg.truth_loss_weight * truth_loss
    )
    return {
        "loss": total,
        "annotation_loss": annotation_loss,
        "truth_loss": truth_loss,
        "output": output,
        "split": split,
        "num_audit_edges": int(audit_indices.numel()),
    }
