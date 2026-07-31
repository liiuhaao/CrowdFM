from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ConfusionPriorMode = Literal["uniform", "global"]


@dataclass(frozen=True)
class ResidualResult:
    residual: torch.Tensor
    numerator: torch.Tensor
    variance: torch.Tensor
    pair_counts: torch.Tensor
    statistic: float


def _expected_confusion_counts(
    triple: torch.Tensor,
    task_posterior: torch.Tensor,
    *,
    num_worker: int,
    num_option: int,
    edge_mask: torch.Tensor,
) -> torch.Tensor:
    """Accumulate posterior expected worker/class/report counts."""

    if task_posterior.ndim != 2 or task_posterior.shape[1] != num_option:
        raise ValueError("task_posterior has an incompatible option dimension")
    if edge_mask.shape != (triple.shape[1],):
        raise ValueError("edge_mask must have one entry per annotation edge")

    device = task_posterior.device
    dtype = task_posterior.dtype
    triple = triple.to(device)
    edge_mask = edge_mask.to(device)

    counts = torch.zeros(
        (num_worker, num_option, num_option),
        dtype=dtype,
        device=device,
    )
    worker_ids = triple[0, edge_mask].long()
    reported = triple[1, edge_mask].long()
    task_ids = triple[2, edge_mask].long()
    expected_truth = task_posterior[task_ids]

    for answer in range(num_option):
        answer_mask = reported == answer
        if torch.any(answer_mask):
            counts[:, :, answer].index_add_(
                0,
                worker_ids[answer_mask],
                expected_truth[answer_mask],
            )
    return counts


def confusion_posterior_parameters(
    triple: torch.Tensor,
    task_posterior: torch.Tensor,
    *,
    num_worker: int,
    num_option: int,
    edge_mask: torch.Tensor,
    prior_strength: float = 1.0,
    prior_mode: ConfusionPriorMode = "uniform",
) -> torch.Tensor:
    """Return Dirichlet posterior parameters for worker confusion rows.

    ``uniform`` uses a symmetric Dirichlet prior. ``global`` uses a
    leave-one-worker-out pooled class-conditional confusion row as the prior
    mean. The latter stabilizes sparse multiclass workers without sharing their
    own labels twice.
    """

    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    if prior_mode not in ("uniform", "global"):
        raise ValueError("prior_mode must be 'uniform' or 'global'")

    counts = _expected_confusion_counts(
        triple,
        task_posterior,
        num_worker=num_worker,
        num_option=num_option,
        edge_mask=edge_mask,
    )

    if prior_mode == "uniform":
        prior = torch.full_like(counts, prior_strength / num_option)
        return counts + prior

    total_counts = counts.sum(dim=0, keepdim=True)
    leave_one_out = (total_counts - counts).clamp_min(0.0)
    smoothing = torch.full_like(leave_one_out, 1.0 / num_option)
    global_rows = leave_one_out + smoothing
    global_rows = global_rows / global_rows.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(global_rows.dtype).tiny
    )
    return counts + prior_strength * global_rows


def estimate_confusion_matrices(
    triple: torch.Tensor,
    task_posterior: torch.Tensor,
    *,
    num_worker: int,
    num_option: int,
    edge_mask: torch.Tensor,
    prior_strength: float = 1.0,
    prior_mode: ConfusionPriorMode = "uniform",
) -> torch.Tensor:
    """Estimate P_i(reported=a | truth=c) by posterior expected counts."""

    counts = confusion_posterior_parameters(
        triple,
        task_posterior,
        num_worker=num_worker,
        num_option=num_option,
        edge_mask=edge_mask,
        prior_strength=prior_strength,
        prior_mode=prior_mode,
    )
    return counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def item_conditioned_disagreement(
    task_posterior: torch.Tensor,
    confusion: torch.Tensor,
    worker_i: torch.Tensor,
    worker_j: torch.Tensor,
    task_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute item-conditioned model-implied pairwise disagreement."""

    p_i = confusion[worker_i.long()]
    p_j = confusion[worker_j.long()]
    agreement_given_class = (p_i * p_j).sum(dim=-1)
    agreement = (task_posterior[task_ids.long()] * agreement_given_class).sum(dim=-1)
    return 1.0 - agreement


def spectral_statistic(residual: torch.Tensor) -> float:
    """Return the two-sided spectral statistic ||R||_op for symmetric R."""

    if residual.ndim != 2 or residual.shape[0] != residual.shape[1]:
        raise ValueError("residual must be a square matrix")
    symmetric = 0.5 * (residual + residual.transpose(0, 1))
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    return float(eigenvalues.abs().max().item())


def build_residual_matrix(
    triple: torch.Tensor,
    task_posterior: torch.Tensor,
    confusion: torch.Tensor,
    *,
    audit_edge_mask: torch.Tensor,
    min_pair_count: int = 1,
    probability_clip: float = 1e-4,
    variance_ridge: float = 1e-8,
) -> ResidualResult:
    """Construct the signed, standardized worker-pair residual matrix."""

    if min_pair_count < 1:
        raise ValueError("min_pair_count must be at least one")
    if not 0.0 < probability_clip < 0.5:
        raise ValueError("probability_clip must lie in (0, 0.5)")

    device = task_posterior.device
    dtype = task_posterior.dtype
    triple = triple.to(device)
    audit_edge_mask = audit_edge_mask.to(device)

    num_worker = confusion.shape[0]
    numerator = torch.zeros((num_worker, num_worker), dtype=dtype, device=device)
    variance = torch.zeros_like(numerator)
    pair_counts = torch.zeros((num_worker, num_worker), dtype=torch.long, device=device)

    audit_tasks = torch.unique(triple[2, audit_edge_mask].long())
    for task in audit_tasks.tolist():
        edge_indices = torch.nonzero(
            audit_edge_mask & (triple[2].long() == task), as_tuple=False
        ).flatten()
        if edge_indices.numel() < 2:
            continue

        local_pairs = torch.triu_indices(
            edge_indices.numel(), edge_indices.numel(), offset=1, device=device
        )
        left_edges = edge_indices[local_pairs[0]]
        right_edges = edge_indices[local_pairs[1]]
        worker_i = triple[0, left_edges].long()
        worker_j = triple[0, right_edges].long()
        distinct_worker = worker_i != worker_j
        if not torch.any(distinct_worker):
            continue

        worker_i = worker_i[distinct_worker]
        worker_j = worker_j[distinct_worker]
        left_edges = left_edges[distinct_worker]
        right_edges = right_edges[distinct_worker]
        task_vector = torch.full_like(worker_i, task)

        predicted = item_conditioned_disagreement(
            task_posterior,
            confusion,
            worker_i,
            worker_j,
            task_vector,
        ).clamp(probability_clip, 1.0 - probability_clip)
        observed = (triple[1, left_edges] != triple[1, right_edges]).to(dtype)
        delta = observed - predicted
        cell_variance = predicted * (1.0 - predicted)
        ones = torch.ones_like(worker_i, dtype=torch.long)

        numerator.index_put_((worker_i, worker_j), delta, accumulate=True)
        numerator.index_put_((worker_j, worker_i), delta, accumulate=True)
        variance.index_put_((worker_i, worker_j), cell_variance, accumulate=True)
        variance.index_put_((worker_j, worker_i), cell_variance, accumulate=True)
        pair_counts.index_put_((worker_i, worker_j), ones, accumulate=True)
        pair_counts.index_put_((worker_j, worker_i), ones, accumulate=True)

    residual = torch.zeros_like(numerator)
    supported = pair_counts >= min_pair_count
    residual[supported] = numerator[supported] / torch.sqrt(
        variance[supported] + variance_ridge
    )
    residual.fill_diagonal_(0.0)

    return ResidualResult(
        residual=residual,
        numerator=numerator,
        variance=variance,
        pair_counts=pair_counts,
        statistic=spectral_statistic(residual),
    )
