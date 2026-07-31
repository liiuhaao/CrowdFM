from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from cfm.audit.pipeline import masked_data
from cfm.audit.predictive_split import make_annotation_audit_split
from cfm.data.crowd_data import CrowdData

from .likelihood import (
    diagonal_gaussian_kl,
    joint_annotation_log_likelihood,
    supervised_emission_loss,
    symmetric_gaussian_kl,
)


@dataclass(frozen=True)
class CrowdSITrainingConfig:
    audit_task_fraction: float = 1.0
    context_fraction: float = 0.5
    min_context_workers: int = 1
    min_audit_workers: int = 2
    min_worker_context_edges: int = 0
    truth_loss_weight: float = 1.0
    annotation_loss_weight: float = 1.0
    assignment_loss_weight: float = 0.25
    mechanism_kl_weight: float = 1e-3
    consistency_loss_weight: float = 0.1
    assignment_negative_ratio: float = 1.0


def _sample_negative_pairs(
    data: CrowdData,
    num_negative: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    workers = data.triple[0].detach().cpu().long()
    tasks = data.triple[2].detach().cpu().long()
    observed = torch.zeros(data.num_worker * data.num_task, dtype=torch.bool)
    observed[workers * data.num_task + tasks] = True
    candidates = torch.nonzero(~observed, as_tuple=False).flatten()
    total_negative = int(candidates.numel())
    if num_negative < 1 or total_negative == 0:
        empty = torch.empty(0, dtype=torch.long, device=data.triple.device)
        return empty, empty, total_negative
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = candidates[
        torch.randperm(candidates.numel(), generator=generator)[:num_negative]
    ]
    negative_workers = torch.div(selected, data.num_task, rounding_mode="floor")
    negative_tasks = selected.remainder(data.num_task)
    return (
        negative_workers.to(data.triple.device),
        negative_tasks.to(data.triple.device),
        total_negative,
    )


def _assignment_logits_from_output(
    model: torch.nn.Module,
    output: dict[str, torch.Tensor],
    workers: torch.Tensor,
    tasks: torch.Tensor,
) -> torch.Tensor:
    workers = workers.to(output["z_worker_si"].device, dtype=torch.long)
    tasks = tasks.to(output["z_task_si"].device, dtype=torch.long)
    features = torch.cat(
        [
            output["z_worker_si"][workers],
            output["z_task_si"][tasks],
            output["mechanism_context"].expand(workers.numel(), -1),
        ],
        dim=-1,
    )
    return model.assignment_head(features).squeeze(-1)


def crowdsi_training_loss(
    model: torch.nn.Module,
    data: CrowdData,
    *,
    config: CrowdSITrainingConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Train aggregation, response generation, assignment, and system ID jointly."""

    cfg = config or CrowdSITrainingConfig()
    nonnegative = [
        cfg.truth_loss_weight,
        cfg.annotation_loss_weight,
        cfg.assignment_loss_weight,
        cfg.mechanism_kl_weight,
        cfg.consistency_loss_weight,
        cfg.assignment_negative_ratio,
    ]
    if any(value < 0 for value in nonnegative) or cfg.annotation_loss_weight == 0:
        raise ValueError("loss weights must be nonnegative and annotation weight positive")

    split = make_annotation_audit_split(
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
    query_tasks = data.triple[2, audit_indices].long()
    observed_answers = data.triple[1, audit_indices].long()
    output = model(
        context_data,
        query_workers=query_workers,
        query_tasks=query_tasks,
        sample_mechanism=True,
    )

    task_y = getattr(data, "task_y", None)
    known_truth = isinstance(task_y, torch.Tensor) and torch.any(task_y >= 0)
    if known_truth:
        annotation_loss = supervised_emission_loss(
            output["hat_annotation_given_truth"],
            observed_answers,
            query_tasks,
            task_y,
        )
    else:
        annotation_loss = -joint_annotation_log_likelihood(
            output["hat_task_option"],
            output["hat_annotation_given_truth"],
            observed_answers,
            query_tasks,
            reduction="mean",
        )

    truth_loss = annotation_loss.new_zeros(())
    if cfg.truth_loss_weight > 0 and isinstance(task_y, torch.Tensor):
        task_y = task_y.to(output["hat_task_option"].device, dtype=torch.long)
        valid = task_y >= 0
        if torch.any(valid):
            truth_loss = F.cross_entropy(output["hat_task_option"][valid], task_y[valid])

    positive_assignment_logits = output["hat_assignment_logit"]
    num_negative = int(round(cfg.assignment_negative_ratio * audit_indices.numel()))
    negative_workers, negative_tasks, total_negative = _sample_negative_pairs(
        data,
        num_negative,
        seed=seed + 1,
    )
    total_positive = int(data.triple.shape[1])
    positive_sampling_weight = total_positive / max(1, positive_assignment_logits.numel())
    positive_loss_sum = (
        positive_sampling_weight * F.softplus(-positive_assignment_logits).sum()
    )
    if negative_workers.numel() > 0:
        negative_logits = _assignment_logits_from_output(
            model,
            output,
            negative_workers,
            negative_tasks,
        )
        negative_sampling_weight = total_negative / negative_logits.numel()
        negative_loss_sum = (
            negative_sampling_weight * F.softplus(negative_logits).sum()
        )
    else:
        negative_loss_sum = positive_loss_sum.new_zeros(())
    assignment_denominator = max(1, total_positive + total_negative)
    assignment_loss = (positive_loss_sum + negative_loss_sum) / assignment_denominator

    second_split = make_annotation_audit_split(
        data.triple,
        data.num_task,
        num_worker=data.num_worker,
        audit_task_fraction=cfg.audit_task_fraction,
        context_fraction=cfg.context_fraction,
        min_context_workers=cfg.min_context_workers,
        min_audit_workers=cfg.min_audit_workers,
        min_worker_context_edges=cfg.min_worker_context_edges,
        seed=seed + 2,
    )
    second_context = masked_data(data, second_split.context_edge_mask)
    second_output = model(second_context, sample_mechanism=False)
    consistency_loss = symmetric_gaussian_kl(
        output["mechanism_mean"],
        output["mechanism_log_variance"],
        second_output["mechanism_mean"],
        second_output["mechanism_log_variance"],
    ) / output["mechanism_mean"].numel()
    mechanism_kl = diagonal_gaussian_kl(
        output["mechanism_mean"],
        output["mechanism_log_variance"],
    ) / output["mechanism_mean"].numel()

    total = (
        cfg.annotation_loss_weight * annotation_loss
        + cfg.truth_loss_weight * truth_loss
        + cfg.assignment_loss_weight * assignment_loss
        + cfg.mechanism_kl_weight * mechanism_kl
        + cfg.consistency_loss_weight * consistency_loss
    )
    return {
        "loss": total,
        "annotation_loss": annotation_loss,
        "truth_loss": truth_loss,
        "assignment_loss": assignment_loss,
        "mechanism_kl": mechanism_kl,
        "consistency_loss": consistency_loss,
        "output": output,
        "split": split,
        "num_audit_edges": int(audit_indices.numel()),
        "num_assignment_negatives": total_negative,
    }
