from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PredictiveResidualResult:
    marginal_statistic: float
    dependence_statistic: float
    residual_matrix: torch.Tensor
    pair_counts: torch.Tensor
    standardized_edge_residuals: torch.Tensor
    marginal_probabilities: torch.Tensor
    emission_probabilities: torch.Tensor
    task_posterior: torch.Tensor


def normalize_annotation_probabilities(
    probabilities: torch.Tensor,
    *,
    probability_clip: float = 1e-4,
) -> torch.Tensor:
    if probabilities.ndim < 2:
        raise ValueError("probabilities must have a final option axis")
    if probabilities.shape[-1] < 2:
        raise ValueError("at least two response options are required")
    if probability_clip < 0.0 or probability_clip >= 0.5:
        raise ValueError("probability_clip must lie in [0, 0.5)")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    normalized = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    if probability_clip > 0.0:
        normalized = normalized.clamp_min(probability_clip)
        normalized = normalized / normalized.sum(dim=-1, keepdim=True)
    return normalized


def marginal_response_probabilities(
    task_posterior: torch.Tensor,
    emission_probabilities: torch.Tensor,
    query_tasks: torch.Tensor,
) -> torch.Tensor:
    """Integrate the edge-conditioned emission law over shared task truth."""

    if task_posterior.ndim != 2:
        raise ValueError("task_posterior must have shape [num_tasks, num_options]")
    if emission_probabilities.ndim != 3:
        raise ValueError(
            "emission_probabilities must have shape [num_edges, num_truth, num_report]"
        )
    if emission_probabilities.shape[1:] != (
        task_posterior.shape[1],
        task_posterior.shape[1],
    ):
        raise ValueError("truth and report option dimensions must match task posterior")
    query_tasks = query_tasks.to(task_posterior.device, dtype=torch.long)
    return torch.einsum(
        "ec,eca->ea",
        task_posterior[query_tasks],
        emission_probabilities,
    )


def standardized_categorical_residuals(
    probabilities: torch.Tensor,
    observed_answers: torch.Tensor,
    *,
    variance_ridge: float = 1e-8,
) -> torch.Tensor:
    observed_answers = observed_answers.to(probabilities.device, dtype=torch.long)
    one_hot = F.one_hot(observed_answers, probabilities.shape[-1]).to(probabilities.dtype)
    return (one_hot - probabilities) / torch.sqrt(
        probabilities * (1.0 - probabilities) + variance_ridge
    )


def marginal_categorical_statistic(
    probabilities: torch.Tensor,
    observed_answers: torch.Tensor,
    *,
    variance_ridge: float = 1e-8,
) -> torch.Tensor:
    observed_answers = observed_answers.to(probabilities.device, dtype=torch.long)
    one_hot = F.one_hot(observed_answers, probabilities.shape[-1]).to(probabilities.dtype)
    centered = one_hot - probabilities
    variance = (probabilities * (1.0 - probabilities)).sum(dim=0) + variance_ridge
    return torch.linalg.vector_norm(centered.sum(dim=0) / torch.sqrt(variance), ord=2)


def spectral_statistic(residual_matrix: torch.Tensor) -> float:
    if residual_matrix.ndim != 2 or residual_matrix.shape[0] != residual_matrix.shape[1]:
        raise ValueError("residual_matrix must be square")
    if residual_matrix.numel() == 0:
        return 0.0
    return float(torch.linalg.eigvalsh(residual_matrix).abs().max().item())


def build_predictive_residual(
    task_posterior: torch.Tensor,
    emission_probabilities: torch.Tensor,
    observed_answers: torch.Tensor,
    query_workers: torch.Tensor,
    query_tasks: torch.Tensor,
    *,
    num_worker: int,
    min_pair_count: int = 1,
    probability_clip: float = 1e-4,
    variance_ridge: float = 1e-8,
) -> PredictiveResidualResult:
    """Build marginal and pairwise-disagreement residual statistics.

    The learned model supplies q_k(c) and edge-conditioned emissions
    P_e(a|c).  Marginal response probabilities are obtained by integrating over
    q_k.  For each pair of held-out workers on the same task, the shared latent
    truth induces the model-implied disagreement probability

        1 - sum_c q_k(c) sum_a P_i(a|c) P_j(a|c).

    This preserves legitimate within-task dependence from task-truth uncertainty;
    only residual disagreement beyond that law enters the spectral matrix.
    """

    if num_worker < 1 or min_pair_count < 1 or variance_ridge <= 0:
        raise ValueError("worker count, pair count, and variance ridge must be positive")
    if query_workers.ndim != 1 or query_tasks.ndim != 1:
        raise ValueError("query_workers and query_tasks must be one-dimensional")
    if query_workers.shape != query_tasks.shape:
        raise ValueError("query_workers and query_tasks must have the same shape")
    if emission_probabilities.shape[0] != query_workers.numel():
        raise ValueError("one emission tensor is required for every query edge")

    task_posterior = normalize_annotation_probabilities(
        task_posterior, probability_clip=probability_clip
    )
    emission_probabilities = normalize_annotation_probabilities(
        emission_probabilities, probability_clip=probability_clip
    )
    query_workers = query_workers.to(task_posterior.device, dtype=torch.long)
    query_tasks = query_tasks.to(task_posterior.device, dtype=torch.long)
    observed_answers = observed_answers.to(task_posterior.device, dtype=torch.long)

    marginal_probabilities = marginal_response_probabilities(
        task_posterior, emission_probabilities, query_tasks
    )
    edge_residuals = standardized_categorical_residuals(
        marginal_probabilities,
        observed_answers,
        variance_ridge=variance_ridge,
    )
    marginal = marginal_categorical_statistic(
        marginal_probabilities,
        observed_answers,
        variance_ridge=variance_ridge,
    )

    residual_sum = torch.zeros(
        (num_worker, num_worker), dtype=task_posterior.dtype, device=task_posterior.device
    )
    variance_sum = torch.zeros_like(residual_sum)
    pair_counts = torch.zeros(
        (num_worker, num_worker), dtype=torch.long, device=task_posterior.device
    )

    for task in torch.unique(query_tasks).tolist():
        indices = torch.nonzero(query_tasks == task, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        pairs = torch.triu_indices(
            indices.numel(), indices.numel(), offset=1, device=task_posterior.device
        )
        left, right = indices[pairs[0]], indices[pairs[1]]
        worker_i, worker_j = query_workers[left], query_workers[right]
        distinct = worker_i != worker_j
        left, right = left[distinct], right[distinct]
        worker_i, worker_j = worker_i[distinct], worker_j[distinct]
        if left.numel() == 0:
            continue

        agreement_by_truth = (
            emission_probabilities[left] * emission_probabilities[right]
        ).sum(dim=-1)
        predicted_disagreement = 1.0 - (
            agreement_by_truth * task_posterior[task].unsqueeze(0)
        ).sum(dim=-1)
        predicted_disagreement = predicted_disagreement.clamp(
            probability_clip, 1.0 - probability_clip
        )
        observed_disagreement = (observed_answers[left] != observed_answers[right]).to(
            task_posterior.dtype
        )
        delta = observed_disagreement - predicted_disagreement
        variance = predicted_disagreement * (1.0 - predicted_disagreement)

        residual_sum.index_put_((worker_i, worker_j), delta, accumulate=True)
        residual_sum.index_put_((worker_j, worker_i), delta, accumulate=True)
        variance_sum.index_put_((worker_i, worker_j), variance, accumulate=True)
        variance_sum.index_put_((worker_j, worker_i), variance, accumulate=True)
        ones = torch.ones_like(worker_i, dtype=torch.long)
        pair_counts.index_put_((worker_i, worker_j), ones, accumulate=True)
        pair_counts.index_put_((worker_j, worker_i), ones, accumulate=True)

    supported = pair_counts >= min_pair_count
    residual_matrix = torch.zeros_like(residual_sum)
    residual_matrix[supported] = residual_sum[supported] / torch.sqrt(
        variance_sum[supported] + variance_ridge
    )
    residual_matrix.fill_diagonal_(0.0)

    return PredictiveResidualResult(
        marginal_statistic=float(marginal.item()),
        dependence_statistic=spectral_statistic(residual_matrix),
        residual_matrix=residual_matrix,
        pair_counts=pair_counts,
        standardized_edge_residuals=edge_residuals,
        marginal_probabilities=marginal_probabilities,
        emission_probabilities=emission_probabilities,
        task_posterior=task_posterior,
    )
